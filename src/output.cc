#include <boost/algorithm/string/predicate.hpp>
#include "output.h"
BOOST_CLASS_EXPORT_IMPLEMENT(SoftmaxOutputModel)
BOOST_CLASS_EXPORT_IMPLEMENT(MlpSoftmaxOutputModel)
BOOST_CLASS_EXPORT_IMPLEMENT(MorphologyOutputModel)
BOOST_CLASS_EXPORT_IMPLEMENT(RnngOutputModel)

const unsigned lstm_layer_count = 2;

OutputModel::~OutputModel() {}

bool OutputModel::IsDone() const {
  return IsDone(GetStatePointer());
}

Expression OutputModel::GetState() const {
  return GetState(GetStatePointer());
}

SoftmaxOutputModel::SoftmaxOutputModel() : fsb(nullptr) {}

SoftmaxOutputModel::SoftmaxOutputModel(Model& model, unsigned embedding_dim, unsigned context_dim, unsigned state_dim, Dict* vocab, const string& clusters_filename) : state_dim(state_dim) {
  if (clusters_filename.length() > 0) {
    fsb = new ClassFactoredSoftmaxBuilder(state_dim, clusters_filename, vocab, &model);
  }
  else {
    fsb = new StandardSoftmaxBuilder(state_dim, vocab->size(), &model);
  }
  embeddings = model.add_lookup_parameters(vocab->size(), {embedding_dim});
  output_builder = LSTMBuilder(lstm_layer_count, embedding_dim + context_dim, state_dim, &model);
  p_output_builder_initial_state = model.add_parameters({lstm_layer_count * 2 * state_dim});

  kEOS = vocab->convert("</s>");
}

void SoftmaxOutputModel::NewGraph(ComputationGraph& cg) {
  output_builder.new_graph(cg);
  output_builder_initial_state = parameter(cg, p_output_builder_initial_state);
  vector<Expression> h0 = MakeLSTMInitialState(output_builder_initial_state, state_dim, output_builder.layers);
  output_builder.start_new_sequence(h0);
  fsb->new_graph(cg);
  pcg = &cg;
  done.clear();
}

void SoftmaxOutputModel::SetDropout(float rate) {
  //output_builder.set_dropout(rate);
}

Expression SoftmaxOutputModel::GetState(RNNPointer p) const {
  if (p == -1) {
    if (output_builder.h0.size() == 0) {
      return zeroes(*pcg, {state_dim});
    }
    else {
      return output_builder.back();
    }
  }
  return output_builder.get_h(p).back();
}

RNNPointer SoftmaxOutputModel::GetStatePointer() const {
  return output_builder.state();
}

Expression SoftmaxOutputModel::AddInput(const Word* prev_word, const Expression& context) {
  return AddInput(prev_word, context, output_builder.state());
}

Expression SoftmaxOutputModel::Embed(const StandardWord* word) {
  return lookup(*pcg, embeddings, word->id);
}

Expression SoftmaxOutputModel::AddInput(const Word* prev_word_, const Expression& context, const RNNPointer& p) {
  const StandardWord* prev_word = dynamic_cast<const StandardWord*>(prev_word_);
  done.push_back(prev_word->id == kEOS);
  Expression prev_embedding = Embed(prev_word);
  Expression input = concatenate({prev_embedding, context});
  Expression state = output_builder.add_input(p, input);
  assert (done.size() == (size_t)output_builder.state() + 1);
  return state;
}

Expression SoftmaxOutputModel::PredictLogDistribution(const Expression& state) {
  return fsb->full_log_distribution(state);
}

KBestList<Word*> SoftmaxOutputModel::PredictKBest(const Expression& state, unsigned K) {
  // TODO: Manange memory better. Don't just create a bajillion new words and then never delete them
  vector<float> dist = as_vector(PredictLogDistribution(state).value());
  KBestList<Word*> kbest(K);
  for (unsigned i = 0; i < dist.size(); ++i) {
    kbest.add(dist[i], new StandardWord(i));
  }
  return kbest;
}

Expression SoftmaxOutputModel::Loss(const Expression& state, const Word* const ref) {
  const StandardWord* r = dynamic_cast<const StandardWord*>(ref);
  return fsb->neg_log_softmax(state, r->id);
}

Word* SoftmaxOutputModel::Sample(const Expression& state) {
  return new StandardWord(fsb->sample(state));
}

bool SoftmaxOutputModel::IsDone(RNNPointer p) const {
  if (p == -1) {
    return false;
  }

  assert (p < done.size());
  return done[p];
}

MlpSoftmaxOutputModel::MlpSoftmaxOutputModel() {}

MlpSoftmaxOutputModel::MlpSoftmaxOutputModel(Model& model, unsigned embedding_dim, unsigned context_dim, unsigned state_dim, unsigned hidden_dim, Dict* vocab, const string& clusters_filename) : SoftmaxOutputModel(model, embedding_dim, context_dim, state_dim, vocab, clusters_filename) {
  p_W = model.add_parameters({hidden_dim, state_dim});
  p_b = model.add_parameters({hidden_dim});
}

void MlpSoftmaxOutputModel::NewGraph(ComputationGraph& cg) {
  SoftmaxOutputModel::NewGraph(cg);
  W = parameter(cg, p_W);
  b = parameter(cg, p_b);
}

Expression MlpSoftmaxOutputModel::PredictLogDistribution(const Expression& state) {
  Expression h = tanh(affine_transform({b, W, state}));
  return SoftmaxOutputModel::PredictLogDistribution(h);
}

Expression MlpSoftmaxOutputModel::Loss(const Expression& state, const Word* const ref) {
  Expression h = tanh(affine_transform({b, W, state}));
  return SoftmaxOutputModel::Loss(h, ref);
}

Word* MlpSoftmaxOutputModel::Sample(const Expression& state) {
  Expression h = tanh(affine_transform({b, W, state}));
  return SoftmaxOutputModel::Sample(h);
}

MorphologyOutputModel::MorphologyOutputModel() {}

MorphologyOutputModel::MorphologyOutputModel(Model& model, Dict& word_vocab, Dict& root_vocab, unsigned affix_vocab_size, unsigned char_vocab_size, unsigned word_emb_dim, unsigned root_emb_dim, unsigned affix_emb_dim, unsigned char_emb_dim, unsigned model_chooser_hidden_dim, unsigned affix_init_hidden_dim, unsigned char_init_hidden_dim, unsigned state_dim, unsigned affix_lstm_dim, unsigned char_lstm_dim, unsigned context_dim, const string& word_clusters, const string& root_clusters) : state_dim(state_dim), affix_lstm_dim(affix_lstm_dim), char_lstm_dim(char_lstm_dim), pcg(nullptr) {
  unsigned mode_count = 4; // EOS, char, morph, word
  model_chooser = MLP(model, context_dim, model_chooser_hidden_dim, mode_count);

  // We first use the context to predict a root, then use the root+context to initialize the affix LSTM
  affix_lstm_init = MLP(model, context_dim + root_emb_dim, affix_init_hidden_dim, lstm_layer_count * affix_lstm_dim);

  // The char LSTM is initialized just from the context
  char_lstm_init = MLP(model, context_dim, char_init_hidden_dim, lstm_layer_count * char_lstm_dim);

  affix_lstm = LSTMBuilder(lstm_layer_count, affix_emb_dim, affix_lstm_dim, &model);
  char_lstm = LSTMBuilder(lstm_layer_count, char_emb_dim, char_lstm_dim, &model);

  // The main LSTM takes in the context as well as the char, affix, and word embeddings of the current word. The state is included implicitly.
  output_builder = LSTMBuilder(lstm_layer_count, char_lstm_dim + affix_lstm_dim + word_emb_dim + context_dim, state_dim, &model);
  output_lstm_init = model.add_parameters({state_dim * lstm_layer_count});
  embedder = MorphologyEmbedder(model, word_vocab.size(), root_vocab.size(), affix_vocab_size, char_vocab_size, word_emb_dim, affix_emb_dim, char_emb_dim, affix_lstm_dim, char_lstm_dim);

  // Unlike on the encoder side, we need a separate root_emb_dim here. This is because we want to use three things to initialize
  // the affix lstm: the state, the context, and the root. On the input side we have root_emb_dim = lstm_layer_count * affix_lstm_dim
  // but here we would need root_emb_dim = lstm_layer_count * (affix_lstm_dim + state_dim + context_dim) which is far too much,
  // so instead we add an affine transform to transform the larger amount of available information into something manageable.
  root_embeddings = model.add_lookup_parameters(root_vocab.size(), {root_emb_dim});
  affix_embeddings = model.add_lookup_parameters(affix_vocab_size, {affix_emb_dim});
  char_embeddings = model.add_lookup_parameters(char_vocab_size, {char_emb_dim});

  if (word_clusters.length() > 0) {
    word_softmax = new ClassFactoredSoftmaxBuilder(state_dim, word_clusters, &word_vocab, &model);
  }
  else {
    word_softmax = new StandardSoftmaxBuilder(state_dim, word_vocab.size(), &model);
  }
  if (root_clusters.length() > 0) {
    root_softmax = new ClassFactoredSoftmaxBuilder(state_dim, word_clusters, &root_vocab, &model);
  }
  else {
    root_softmax = new StandardSoftmaxBuilder(state_dim, root_vocab.size(), &model);
  }
  affix_softmax = new StandardSoftmaxBuilder(affix_lstm_dim, affix_vocab_size, &model);
  char_softmax = new StandardSoftmaxBuilder(char_lstm_dim, char_vocab_size, &model);
}

void MorphologyOutputModel::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
  model_chooser.NewGraph(cg);
  affix_lstm_init.NewGraph(cg);
  char_lstm_init.NewGraph(cg);
  affix_lstm.new_graph(cg);
  char_lstm.new_graph(cg);
  output_builder.new_graph(cg);
  embedder.NewGraph(cg);
  word_softmax->new_graph(cg);
  root_softmax->new_graph(cg);
  affix_softmax->new_graph(cg);
  char_softmax->new_graph(cg);

  Expression output_lstm_init_expr = parameter(cg, output_lstm_init);
  output_lstm_init_v = MakeLSTMInitialState(output_lstm_init_expr, state_dim, output_builder.layers);
  output_builder.start_new_sequence(output_lstm_init_v);
}

void MorphologyOutputModel::SetDropout(float rate) {}

Expression MorphologyOutputModel::GetState(RNNPointer p) const {
  if (p == -1) {
    if (output_builder.h0.size() == 0) {
      return zeroes(*pcg, {state_dim});
    }
    else {
      return output_builder.back();
    }
  }
  return output_builder.get_h(p).back();
}

RNNPointer MorphologyOutputModel::GetStatePointer() const {
  return output_builder.state();
}

Expression MorphologyOutputModel::AddInput(const Word* const prev_word, const Expression& context) {
  return AddInput(prev_word, context, output_builder.state());
}

Expression MorphologyOutputModel::AddInput(const Word* const prev_word, const Expression& context, const RNNPointer& p) {
  Expression prev_embedding = embedder.Embed(prev_word);
  Expression input = concatenate({prev_embedding, context});
  Expression state = output_builder.add_input(p, input);
  return state;
}

Expression MorphologyOutputModel::PredictLogDistribution(const Expression& state) {
  assert (false);
}

KBestList<Word*> MorphologyOutputModel::PredictKBest(const Expression& state, unsigned K) {
  assert (false);
}

Word* MorphologyOutputModel::Sample(const Expression& state) {
  assert (false);
}

Expression MorphologyOutputModel::WordLoss(const Expression& state, const WordId ref) {
  return word_softmax->neg_log_softmax(state, ref);
}

Expression MorphologyOutputModel::AnalysisLoss(const Expression& state, const Analysis& ref) {
  Expression root_loss = root_softmax->neg_log_softmax(state, ref.root);
  if (ref.affixes.size() == 0) {
    return root_loss;
  }

  Expression root_emb = lookup(*pcg, root_embeddings, ref.root);
  Expression mlp_input = concatenate({state, root_emb});
  Expression affix_lstm_init_expr = affix_lstm_init.Feed(mlp_input);
  vector<Expression> affix_lstm_init_v = MakeLSTMInitialState(affix_lstm_init_expr, affix_lstm_dim, affix_lstm.layers);
  affix_lstm.start_new_sequence(affix_lstm_init_v);

  vector<Expression> affix_losses;
  for (WordId affix : ref.affixes) {
    Expression affix_loss = affix_softmax->neg_log_softmax(affix_lstm.back(), affix);
    Expression affix_emb = lookup(*pcg, affix_embeddings, affix);
    affix_lstm.add_input(affix_emb);
    affix_losses.push_back(affix_loss);
  }

  return root_loss + sum(affix_losses);
}

Expression MorphologyOutputModel::MorphLoss(const Expression& state, const vector<Analysis>& ref) {
  assert (ref.size() > 0);
  Expression min_loss = AnalysisLoss(state, ref[0]);
  for (unsigned i = 0; i < ref.size(); ++i) {
    min_loss = min(min_loss, AnalysisLoss(state, ref[i]));
  }
  return min_loss;
}

Expression MorphologyOutputModel::CharLoss(const Expression& state, const vector<WordId>& ref) {
  assert(ref.size() > 0);

  Expression char_lstm_init_expr = char_lstm_init.Feed(state);
  vector<Expression> char_lstm_init_v = MakeLSTMInitialState(char_lstm_init_expr, char_lstm_dim, char_lstm.layers);
  char_lstm.start_new_sequence(char_lstm_init_v);

  vector<Expression> char_losses;
  for (WordId c : ref) {
    Expression char_loss = char_softmax->neg_log_softmax(char_lstm.back(), c);
    Expression char_emb = lookup(*pcg, char_embeddings, c);
    char_lstm.add_input(char_emb);
    char_losses.push_back(char_loss);
  }

  return sum(char_losses);
}

Expression MorphologyOutputModel::Loss(const Expression& state, const Word* const ref) {
  const MorphoWord* r = dynamic_cast<const MorphoWord*>(ref);
  Expression model_probs = log_softmax(model_chooser.Feed(state));
  Expression word_loss = WordLoss(state, r->word);
  Expression morph_loss = MorphLoss(state, r->analyses);
  Expression char_loss = CharLoss(state, r->chars);

  vector<Expression> losses;
  losses.push_back(pick(model_probs, 1) - word_loss);
  losses.push_back(pick(model_probs, 2) - morph_loss);
  losses.push_back(pick(model_probs, 3) - char_loss);
  Expression total_loss = -logsumexp(losses);
  return total_loss;
}

bool MorphologyOutputModel::IsDone(RNNPointer p) const {
  return false;
}

RnngOutputModel::RnngOutputModel() {}

RnngOutputModel::RnngOutputModel(Model& model, unsigned term_emb_dim, unsigned nt_emb_dim, unsigned action_emb_dim, unsigned source_dim, unsigned hidden_dim, Dict* vocab, const string& clusters_file) :
    builder(nullptr), hidden_dim(hidden_dim), pcg(nullptr) {
  assert (vocab != nullptr);

  InitializeDictionaries(*vocab);
  unsigned state_dim = hidden_dim;

  SoftmaxBuilder* fsb = nullptr;
  if (clusters_file.length() > 0) {
    fsb = new ClassFactoredSoftmaxBuilder(state_dim, clusters_file, &term_vocab, &model);
  }
  else {
    fsb = new StandardSoftmaxBuilder(state_dim, term_vocab.size(), &model);
  }

  unsigned action_vocab_size = nt_vocab.size() + 2;
  builder = new SourceConditionedParserBuilder(model, fsb, term_vocab.size(), nt_vocab.size(), action_vocab_size, hidden_dim, term_emb_dim, nt_emb_dim, action_emb_dim, source_dim);
}

void RnngOutputModel::InitializeDictionaries(const Dict& raw_vocab) {
  assert (term_vocab.size() == 0);
  assert (nt_vocab.size() == 0);
  this->raw_vocab = raw_vocab;

  for (WordId i = 0; i < (WordId)raw_vocab.size(); ++i) {
    const string& s = raw_vocab.convert(i);
    assert (s.length() >= 3);
    if (boost::starts_with(s, "NT(")) {
      assert (s[s.length() - 1] == ')');
      string subtype = s.substr(3, s.length() - 4);
      WordId sub_id = nt_vocab.convert(subtype);

      assert (w2a.size() == (unsigned)i);
      Action a = {Action::kNT, sub_id};
      w2a.push_back(a);
      a2w[a] = i;
    }
    else if (boost::starts_with(s, "SHIFT(")) {
      assert (s[s.length() - 1] == ')');
      string subtype = s.substr(6, s.length() - 7);
      WordId sub_id = term_vocab.convert(subtype);

      assert (w2a.size() == (unsigned)i);
      Action a = {Action::kShift, sub_id};
      w2a.push_back(a);
      a2w[a] = i;
    }
    else if (s == "UNK" || s == "<s>" || s == "</s>") {
      assert (false);
      /*assert (w2a.size() == (unsigned)i);
      Action a = {Action::kNone, 0};
      w2a.push_back(a);
      //a2w[a] = i;*/
      continue;
    }
    else {
      assert (s == "REDUCE");
      assert (w2a.size() == (unsigned)i);
      Action a = {Action::kReduce, 0};
      w2a.push_back(a);
      a2w[a] = i;
    }
  }
}

void RnngOutputModel::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
  builder->NewGraph(cg);
  builder->NewSentence();
  source_contexts.clear();
  // TODO: Maybe have an initial context instead of just 0s?
  source_contexts.push_back(zeroes(cg, {hidden_dim}));
}

void RnngOutputModel::SetDropout(float rate) {
  builder->SetDropout(rate);
}

Expression RnngOutputModel::GetState() const {
  assert (source_contexts.size() > 0);
  return builder->GetStateVector(source_contexts.back());
}

Expression RnngOutputModel::GetState(RNNPointer p) const {
  return builder->GetStateVector(source_contexts[p], p);
}

RNNPointer RnngOutputModel::GetStatePointer() const {
  return builder->state();
}

Expression RnngOutputModel::AddInput(const Word* prev_word, const Expression& context) {
  return AddInput(prev_word, context, builder->state());
}

Expression RnngOutputModel::AddInput(const Word* prev_word_, const Expression& context, const RNNPointer& p) {
  const StandardWord* prev_word = dynamic_cast<const StandardWord*>(prev_word_);
  source_contexts.push_back(context);
  Action action = Convert(prev_word->id);
  builder->PerformAction(action, p);
  return builder->GetStateVector(context);
}

Expression RnngOutputModel::PredictLogDistribution(const Expression& source_context) {
  assert (false);
}

KBestList<Word*> RnngOutputModel::PredictKBest(const Expression& source_context, unsigned K) {
  assert (false);
}

Word* RnngOutputModel::Sample(const Expression& source_context) {
  Expression state_vector = builder->GetStateVector(source_context);
  Action action = builder->Sample(state_vector);
  return new StandardWord(Convert(action)); 
}

Expression RnngOutputModel::Loss(const Expression& source_context, const Word* const ref) {
  const StandardWord* r = dynamic_cast<const StandardWord*>(ref);
  Action ref_action = Convert(r->id);
  if (ref_action.type == Action::kNone) {
    return zeroes(*pcg, {1});
  }
  Expression state_vector = builder->GetStateVector(source_context);
  Expression neg_log_prob = builder->Loss(state_vector, ref_action);
  return neg_log_prob;
}

Action RnngOutputModel::Convert(WordId w) const {
  return w2a[w];
}

WordId RnngOutputModel::Convert(const Action& action) const {
  auto it = a2w.find(action);
  assert (it != a2w.end());
  return it->second;
}

bool RnngOutputModel::IsDone(RNNPointer p) const {
  return builder->IsDone(p);
}


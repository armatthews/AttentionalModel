#include <boost/algorithm/string/predicate.hpp>
#include "output.h"
BOOST_CLASS_EXPORT_IMPLEMENT(SoftmaxOutputModel)
BOOST_CLASS_EXPORT_IMPLEMENT(MlpSoftmaxOutputModel)
BOOST_CLASS_EXPORT_IMPLEMENT(MorphologyOutputModel)
BOOST_CLASS_EXPORT_IMPLEMENT(RnngOutputModel)
BOOST_CLASS_EXPORT_IMPLEMENT(DependencyOutputModel)

const unsigned lstm_layer_count = 2;

OutputModel::~OutputModel() {}

bool OutputModel::IsDone() const {
  return IsDone(GetStatePointer());
}

Expression OutputModel::GetState() const {
  return GetState(GetStatePointer());
}

Expression OutputModel::AddInput(const shared_ptr<const Word> prev_word, const Expression& context) {
  return AddInput(prev_word, context, GetStatePointer());
}

Expression OutputModel::PredictLogDistribution(const Expression& state) {
  return PredictLogDistribution(GetStatePointer(), state);
}

KBestList<shared_ptr<Word>> OutputModel::PredictKBest(const Expression& state, unsigned K) {
  return PredictKBest(GetStatePointer(), state, K);
}

pair<shared_ptr<Word>, float> OutputModel::Sample(const Expression& state) {
  return Sample(GetStatePointer(), state);
}

Expression OutputModel::Loss(const Expression& state, const shared_ptr<const Word> ref) {
  return Loss(GetStatePointer(), state, ref);
}

SoftmaxOutputModel::SoftmaxOutputModel() : fsb(nullptr) {}

SoftmaxOutputModel::SoftmaxOutputModel(Model& model, unsigned embedding_dim, unsigned context_dim, unsigned state_dim, Dict* vocab, const string& clusters_filename) : state_dim(state_dim) {
  if (clusters_filename.length() > 0) {
    fsb = new ClassFactoredSoftmaxBuilder(state_dim, clusters_filename, *vocab, model);
  }
  else {
    fsb = new StandardSoftmaxBuilder(state_dim, vocab->size(), model);
  }
  embeddings = model.add_lookup_parameters(vocab->size(), {embedding_dim});
  output_builder = LSTMBuilder(lstm_layer_count, embedding_dim + context_dim, state_dim, model);
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

Expression SoftmaxOutputModel::Embed(const shared_ptr<const StandardWord> word) {
  return lookup(*pcg, embeddings, word->id);
}

Expression SoftmaxOutputModel::AddInput(const shared_ptr<const Word> prev_word_, const Expression& context, const RNNPointer& p) {
  const shared_ptr<const StandardWord> prev_word = dynamic_pointer_cast<const StandardWord>(prev_word_);
  done.push_back(prev_word->id == kEOS);
  Expression prev_embedding = Embed(prev_word);
  Expression input = concatenate({prev_embedding, context});
  Expression state = output_builder.add_input(p, input);
  assert (done.size() == (size_t)output_builder.state() + 1);
  return state;
}

Expression SoftmaxOutputModel::PredictLogDistribution(RNNPointer p, const Expression& state) {
  return fsb->full_log_distribution(state);
}

KBestList<shared_ptr<Word>> SoftmaxOutputModel::PredictKBest(RNNPointer p, const Expression& state, unsigned K) {
  vector<float> dist = as_vector(PredictLogDistribution(p, state).value());
  KBestList<shared_ptr<Word>> kbest(K);
  for (unsigned i = 0; i < dist.size(); ++i) {
    kbest.add(dist[i], make_shared<StandardWord>(i));
  }
  return kbest;
}

Expression max_expr(const vector<Expression>& exprs) {
  assert (exprs.size() > 0);
  Expression M = exprs[0];
  for (unsigned i = 1; i < exprs.size(); ++i) {
    M = max(M, exprs[i]);
  }
  return M;
}

Expression SoftmaxOutputModel::Loss(RNNPointer p, const Expression& state, const shared_ptr<const Word> ref) {
  const shared_ptr<const StandardWord> r = dynamic_pointer_cast<const StandardWord>(ref);

  if (false) {
    unsigned vocab_size = 3118;
    unsigned num_samples = 100;
    Expression scores = dynamic_cast<StandardSoftmaxBuilder*>(fsb)->score(state);
    Expression ref_score = pick(scores, r->id);
    //cerr << "Ref score " << r->id << ": " << as_scalar(ref_score.value()) << endl;
    vector<Expression> sample_scores;
    for (unsigned i = 0; i < num_samples; ++i) {
      unsigned sample_id = rand() % vocab_size;
      Expression sample = pick(scores, sample_id);
      sample_scores.push_back(sample);
      //cerr << "Sample " << i << " score " << sample_id << ": " << as_scalar(sample.value()) << endl;
    }
    Expression max_sample = max_expr(sample_scores);
    max_sample = max(max_sample, ref_score);
    //cerr << "Max sample: " << as_scalar(max_sample.value()) << endl;
    for (unsigned i = 0; i < num_samples; ++i) {
      sample_scores[i] = exp(sample_scores[i] - max_sample);
    }
    float ratio = 1.0f * (vocab_size - 1) / num_samples;
    Expression s = sum(sample_scores) * ratio + exp(ref_score - max_sample);
    Expression denominator = max_sample + log(s);
    Expression log_prob = ref_score - denominator;
    return -log_prob;
  }
  else {
    return fsb->neg_log_softmax(state, r->id);
  }
}

pair<shared_ptr<Word>, float> SoftmaxOutputModel::Sample(RNNPointer p, const Expression& state) {
  unsigned sampled_id = fsb->sample(state);
  shared_ptr<StandardWord> sample = make_shared<StandardWord>(sampled_id);
  Expression score_expr = fsb->neg_log_softmax(state, sampled_id);
  float score = as_scalar(score_expr.value());
  return make_pair(sample, score);
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

Expression MlpSoftmaxOutputModel::GetState(RNNPointer p) const {
  Expression base_state = SoftmaxOutputModel::GetState(p);
  Expression state = tanh(affine_transform({b, W, base_state}));
  return state;
}

Expression MlpSoftmaxOutputModel::AddInput(const shared_ptr<const Word> prev_word_, const Expression& context, const RNNPointer& p) {
  Expression base_state = SoftmaxOutputModel::AddInput(prev_word_, context, p);
  Expression state = tanh(affine_transform({b, W, base_state}));
  return state;
}

void MlpSoftmaxOutputModel::NewGraph(ComputationGraph& cg) {
  SoftmaxOutputModel::NewGraph(cg);
  W = parameter(cg, p_W);
  b = parameter(cg, p_b);
}

Expression MlpSoftmaxOutputModel::AddInput(Expression prev_word_emb, const Expression& context) {
  return AddInput(prev_word_emb, context, GetStatePointer());
}

Expression MlpSoftmaxOutputModel::AddInput(Expression prev_word_emb, const Expression& context, const RNNPointer& p) {
  done.push_back(false);
  Expression input = concatenate({prev_word_emb, context});
  Expression base_state = output_builder.add_input(p, input);
  Expression state = tanh(affine_transform({b, W, base_state}));
  assert (done.size() == (size_t)output_builder.state() + 1);
  return state;
}

MorphologyOutputModel::MorphologyOutputModel() {}

MorphologyOutputModel::MorphologyOutputModel(Model& model, Dict& word_vocab, Dict& root_vocab, unsigned affix_vocab_size, unsigned char_vocab_size, unsigned word_emb_dim, unsigned root_emb_dim, unsigned affix_emb_dim, unsigned char_emb_dim, unsigned model_chooser_hidden_dim, unsigned affix_init_hidden_dim, unsigned char_init_hidden_dim, unsigned state_dim, unsigned affix_lstm_dim, unsigned char_lstm_dim, unsigned context_dim, const string& word_clusters, const string& root_clusters) : state_dim(state_dim), affix_lstm_dim(affix_lstm_dim), char_lstm_dim(char_lstm_dim), pcg(nullptr) {
  const bool use_words = true;
  const bool use_morphology = true;
  unsigned mode_count = 4; // EOS, char, morph, word
  model_chooser = MLP(model, context_dim, model_chooser_hidden_dim, mode_count);

  // We first use the context to predict a root, then use the root+context to initialize the affix LSTM
  affix_lstm_init = MLP(model, context_dim + root_emb_dim, affix_init_hidden_dim, lstm_layer_count * affix_lstm_dim);

  // The char LSTM is initialized just from the context
  char_lstm_init = MLP(model, context_dim, char_init_hidden_dim, lstm_layer_count * char_lstm_dim);

  affix_lstm = LSTMBuilder(lstm_layer_count, affix_emb_dim, affix_lstm_dim, model);
  char_lstm = LSTMBuilder(lstm_layer_count, char_emb_dim, char_lstm_dim, model);

  // The main LSTM takes in the context as well as the char, affix, and word embeddings of the current word. The state is included implicitly.
  output_builder = LSTMBuilder(lstm_layer_count, char_lstm_dim + affix_lstm_dim + word_emb_dim + context_dim, state_dim, model);
  output_lstm_init = model.add_parameters({state_dim * lstm_layer_count});
  embedder = MorphologyEmbedder(model, word_vocab.size(), root_vocab.size(), affix_vocab_size, char_vocab_size, word_emb_dim, affix_emb_dim, char_emb_dim, affix_lstm_dim, char_lstm_dim, use_words, use_morphology);

  // Unlike on the encoder side, we need a separate root_emb_dim here. This is because we want to use three things to initialize
  // the affix lstm: the state, the context, and the root. On the input side we have root_emb_dim = lstm_layer_count * affix_lstm_dim
  // but here we would need root_emb_dim = lstm_layer_count * (affix_lstm_dim + state_dim + context_dim) which is far too much,
  // so instead we add an affine transform to transform the larger amount of available information into something manageable.
  root_embeddings = model.add_lookup_parameters(root_vocab.size(), {root_emb_dim});
  affix_embeddings = model.add_lookup_parameters(affix_vocab_size, {affix_emb_dim});
  char_embeddings = model.add_lookup_parameters(char_vocab_size, {char_emb_dim});

  if (word_clusters.length() > 0) {
    word_softmax = new ClassFactoredSoftmaxBuilder(state_dim, word_clusters, word_vocab, model);
  }
  else {
    word_softmax = new StandardSoftmaxBuilder(state_dim, word_vocab.size(), model);
  }
  if (root_clusters.length() > 0) {
    root_softmax = new ClassFactoredSoftmaxBuilder(state_dim, word_clusters, root_vocab, model);
  }
  else {
    root_softmax = new StandardSoftmaxBuilder(state_dim, root_vocab.size(), model);
  }
  affix_softmax = new StandardSoftmaxBuilder(affix_lstm_dim, affix_vocab_size, model);
  char_softmax = new StandardSoftmaxBuilder(char_lstm_dim, char_vocab_size, model);
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

Expression MorphologyOutputModel::AddInput(const shared_ptr<const Word> prev_word, const Expression& context, const RNNPointer& p) {
  Expression prev_embedding = embedder.Embed(prev_word);
  Expression input = concatenate({prev_embedding, context});
  Expression state = output_builder.add_input(p, input);
  return state;
}

Expression MorphologyOutputModel::PredictLogDistribution(RNNPointer p, const Expression& state) {
  assert (false);
}

KBestList<shared_ptr<Word>> MorphologyOutputModel::PredictKBest(RNNPointer p, const Expression& state, unsigned K) {
  assert (false);
}

pair<shared_ptr<Word>, float> MorphologyOutputModel::Sample(RNNPointer p, const Expression& state) {
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

Expression MorphologyOutputModel::Loss(RNNPointer p, const Expression& state, const shared_ptr<const Word> ref) {
  const shared_ptr<const MorphoWord> r = dynamic_pointer_cast<const MorphoWord>(ref);
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
    fsb = new ClassFactoredSoftmaxBuilder(state_dim, clusters_file, term_vocab, model);
  }
  else {
    fsb = new StandardSoftmaxBuilder(state_dim, term_vocab.size(), model);
  }

  //builder = new FullParserBuilder(model, fsb, term_vocab.size(), nt_vocab.size(), hidden_dim, term_emb_dim, nt_emb_dim, action_emb_dim, source_dim);
  builder = new ParserBuilder(model, fsb, term_vocab.size(), nt_vocab.size(), hidden_dim, term_emb_dim, nt_emb_dim, source_dim);
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
    else if (s == "REDUCE") {
      assert (w2a.size() == (unsigned)i);
      Action a = {Action::kReduce, 0};
      w2a.push_back(a);
      a2w[a] = i;
    }
    else {
      cerr << "Unexpected non-action in input stream: " << s << endl;
      assert (false);
    }
  }
}

void RnngOutputModel::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
  builder->NewGraph(cg);
  builder->NewSentence();
  Expression initial_context = builder->GetInitialContext();
  state_context_vectors.clear();
  state_context_vectors.push_back(builder->GetStateVector(initial_context));
}

void RnngOutputModel::SetDropout(float rate) {
  builder->SetDropout(rate);
}

Expression RnngOutputModel::GetState() const {
  return this->GetState(GetStatePointer());
}

Expression RnngOutputModel::GetState(RNNPointer p) const {
  return state_context_vectors[p];
}

RNNPointer RnngOutputModel::GetStatePointer() const {
  assert ((unsigned)builder->state() == state_context_vectors.size() - 1);
  return builder->state();
}

Expression RnngOutputModel::AddInput(const shared_ptr<const Word> prev_word_, const Expression& context, const RNNPointer& p) {
  const shared_ptr<const StandardWord> prev_word = dynamic_pointer_cast<const StandardWord>(prev_word_);
  Action action = Convert(prev_word->id);
  builder->PerformAction(action, p);
  Expression state_context_vector = builder->GetStateVector(context);
  state_context_vectors.push_back(state_context_vector);
  return state_context_vector;
}

Expression RnngOutputModel::PredictLogDistribution(RNNPointer p, const Expression& source_context) {
  return builder->GetActionDistribution(p, source_context);
}

KBestList<shared_ptr<Word>> RnngOutputModel::PredictKBest(RNNPointer p, const Expression& state_vector, unsigned K) {
  KBestList<Action> kbest_action = builder->PredictKBest(p, state_vector, K);
  KBestList<shared_ptr<Word>> kbest_list(K);
  for (auto score_action : kbest_action.hypothesis_list()) {
    float score = get<0>(score_action);
    Action action = get<1>(score_action);
    kbest_list.add(score, make_shared<StandardWord>(Convert(action)));
  }
  return kbest_list;
}

pair<shared_ptr<Word>, float> RnngOutputModel::Sample(RNNPointer p, const Expression& state_vector) {
  Action action = builder->Sample(p, state_vector);
  unsigned sampled_id = Convert(action);
  shared_ptr<Word> sample = make_shared<StandardWord>(sampled_id);
  Expression score_expr = builder->Loss(p, state_vector, action);
  float score = as_scalar(score_expr.value());
  return make_pair(sample, score);
}

Expression RnngOutputModel::Loss(RNNPointer p, const Expression& state_vector, const shared_ptr<const Word> ref) {
  const shared_ptr<const StandardWord> r = dynamic_pointer_cast<const StandardWord>(ref);
  Action ref_action = Convert(r->id);
  if (ref_action.type == Action::kNone) {
    return zeroes(*pcg, {1});
  }

  Expression neg_log_prob = builder->Loss(p, state_vector, ref_action);
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

DependencyOutputModel::DependencyOutputModel() {}

DependencyOutputModel::DependencyOutputModel(Model& model, Embedder* embedder, unsigned context_dim, unsigned state_dim, unsigned final_hidden_dim, Dict& vocab) {
  assert (state_dim % 2 == 0);
  const unsigned vocab_size = vocab.size();
  half_state_dim = state_dim / 2;

  this->embedder = embedder;
  stack_lstm = LSTMBuilder(lstm_layer_count, half_state_dim + context_dim, half_state_dim, model);
  comp_lstm = LSTMBuilder(lstm_layer_count, half_state_dim + context_dim, half_state_dim, model);
  final_mlp = MLP(model, 2 * half_state_dim, final_hidden_dim, vocab_size);

  emb_transform_p = model.add_parameters({half_state_dim, embedder->Dim()});
  stack_lstm_init_p = model.add_parameters({lstm_layer_count * 2 * half_state_dim});
  comp_lstm_init_p = model.add_parameters({lstm_layer_count * 2 * half_state_dim});

  done_with_left = vocab.convert("</LEFT>");
  done_with_right = vocab.convert("</RIGHT>");
}

void DependencyOutputModel::NewGraph(ComputationGraph& cg) {
  embedder->NewGraph(cg);
  stack_lstm.new_graph(cg);
  comp_lstm.new_graph(cg);
  final_mlp.NewGraph(cg);

  emb_transform = parameter(cg, emb_transform_p);
  stack_lstm_init = MakeLSTMInitialState(parameter(cg, stack_lstm_init_p), half_state_dim, lstm_layer_count);
  comp_lstm_init = MakeLSTMInitialState(parameter(cg, comp_lstm_init_p), half_state_dim, lstm_layer_count);

  stack_lstm.start_new_sequence(stack_lstm_init);
  comp_lstm.start_new_sequence(comp_lstm_init);

  //cerr << "prev_states[" << prev_states.size() << "]: " << stack_lstm.state() << "\t" << comp_lstm.state() << endl;
  prev_states.clear();
  prev_states.push_back(make_tuple(stack_lstm.state(), comp_lstm.state()));

  parent_states.clear();
  parent_states.push_back((RNNPointer)-1);
}

Expression DependencyOutputModel::GetState(RNNPointer p) const {
  RNNPointer stack_pointer;
  RNNPointer comp_pointer;
  tie(stack_pointer, comp_pointer) = prev_states.back();
  Expression stack_state = stack_lstm.get_h(stack_pointer).back();
  Expression comp_state = comp_lstm.get_h(comp_pointer).back();
  return concatenate({stack_state, comp_state});
}

RNNPointer DependencyOutputModel::GetStatePointer() const {
  return (RNNPointer)((int)prev_states.size() - 1);
}

Expression DependencyOutputModel::AddInput(const shared_ptr<const Word> prev_word, const Expression& context, const RNNPointer& p) {
  assert (prev_states.size() == parent_states.size());
  assert (p < prev_states.size());

  unsigned wordid = dynamic_pointer_cast<const StandardWord>(prev_word)->id;
  Expression embedding = embedder->Embed(prev_word);
  Expression transformed_embedding = emb_transform * embedding;

  RNNPointer stack_pointer;
  RNNPointer comp_pointer;
  tie(stack_pointer, comp_pointer) = prev_states[p];

  Expression input_vec = concatenate({transformed_embedding, context});

  if (wordid == done_with_right) {
    RNNPointer old_stack_pointer = stack_pointer;
    stack_pointer = stack_lstm.get_head(stack_pointer);

    Expression node_repr = comp_lstm.add_input(comp_pointer, input_vec);
    RNNPointer parent = parent_states[p];
    RNNPointer grandparent = (parent != (RNNPointer)-1) ? parent_states[parent] : (RNNPointer)-2;

    comp_lstm.add_input(get<1>(prev_states[grandparent]), concatenate({node_repr, context}));
    comp_pointer = comp_lstm.state();
  }
  else if (wordid == done_with_left) {
    comp_lstm.add_input(comp_pointer, input_vec);
    comp_pointer = comp_lstm.state();
  }
  else {
    stack_lstm.add_input(stack_pointer, input_vec);
    comp_lstm.add_input((RNNPointer)-1, input_vec);

    stack_pointer = stack_lstm.state();
    comp_pointer = comp_lstm.state();
  }

  parent_states.push_back(p);
  prev_states.push_back(make_tuple(stack_pointer, comp_pointer));

  assert (prev_states.size() == parent_states.size());
  return OutputModel::GetState();
}

Expression DependencyOutputModel::PredictLogDistribution(RNNPointer p, const Expression& state) {
  assert (false);
}

KBestList<shared_ptr<Word>> DependencyOutputModel::PredictKBest(RNNPointer p, const Expression& state, unsigned K) {
  Expression state2 = GetState(p);
  assert (same_value(state, state2));
  Expression log_probs_expr = final_mlp.Feed(state);
  vector<float> log_probs = as_vector(log_probs_expr.value());

  KBestList<shared_ptr<Word>> kbest;
  for (unsigned i = 0; i < log_probs.size(); ++i) {
    shared_ptr<Word> word = make_shared<StandardWord>(i);
    kbest.add(log_probs[i], word);
  }
  return kbest;
}

pair<shared_ptr<Word>, float> DependencyOutputModel::Sample(RNNPointer p, const Expression& state) {
  assert (false);
}

Expression DependencyOutputModel::Loss(RNNPointer p, const Expression& state, const shared_ptr<const Word> ref) {
  Expression state2 = GetState(p);
  assert (same_value(state, state2));

  Expression log_probs = final_mlp.Feed(state);
  return pickneglogsoftmax(log_probs, dynamic_pointer_cast<const StandardWord>(ref)->id);
}

bool DependencyOutputModel::IsDone(RNNPointer p) const {
  return get<0>(prev_states[p]) == -1;
}

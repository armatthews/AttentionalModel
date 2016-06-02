#include <boost/algorithm/string/predicate.hpp>
#include "output.h"
BOOST_CLASS_EXPORT_IMPLEMENT(SoftmaxOutputModel)
BOOST_CLASS_EXPORT_IMPLEMENT(MlpSoftmaxOutputModel)
//BOOST_CLASS_EXPORT_IMPLEMENT(MorphLmOutputModel)
BOOST_CLASS_EXPORT_IMPLEMENT(RnngOutputModel)

const unsigned lstm_layer_count = 2;

OutputModel::~OutputModel() {}

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
}

void SoftmaxOutputModel::NewGraph(ComputationGraph& cg) {
  output_builder.new_graph(cg);
  output_builder.start_new_sequence();
  fsb->new_graph(cg);
  pcg = &cg;
}

void SoftmaxOutputModel::SetDropout(float rate) {
  //output_builder.set_dropout(rate);
}

Expression SoftmaxOutputModel::GetState() const {
  if (output_builder.state() == -1 && output_builder.h0.size() == 0) {
    return zeroes(*pcg, {state_dim});
  }
  else {
   return output_builder.back();
  }
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

Expression SoftmaxOutputModel::Embed(const Word* w) {
  const StandardWord* word = dynamic_cast<const StandardWord*>(w);
  return lookup(*pcg, embeddings, word->id);
}

Expression SoftmaxOutputModel::AddInput(const Word* prev_word, const Expression& context, const RNNPointer& p) {
  Expression prev_embedding = Embed(prev_word);
  Expression input = concatenate({prev_embedding, context});
  Expression state = output_builder.add_input(p, input);
  return state;
}

Expression SoftmaxOutputModel::PredictLogDistribution(const Expression& state) const {
  return fsb->full_log_distribution(state);
}

Expression SoftmaxOutputModel::Loss(const Expression& state, const Word* const ref) const {
  const StandardWord* r = dynamic_cast<const StandardWord*>(ref);
  return fsb->neg_log_softmax(state, r->id);
}

Word* SoftmaxOutputModel::Sample(const Expression& state) const {
  return new StandardWord(fsb->sample(state));
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

Expression MlpSoftmaxOutputModel::PredictLogDistribution(const Expression& state) const {
  Expression h = tanh(affine_transform({b, W, state}));
  return SoftmaxOutputModel::PredictLogDistribution(h);
}

Expression MlpSoftmaxOutputModel::Loss(const Expression& state, const Word* const ref) const {
  Expression h = tanh(affine_transform({b, W, state}));
  return SoftmaxOutputModel::Loss(h, ref);
}

Word* MlpSoftmaxOutputModel::Sample(const Expression& state) const {
  Expression h = tanh(affine_transform({b, W, state}));
  return SoftmaxOutputModel::Sample(h);
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
    const string& s = raw_vocab.Convert(i);
    assert (s.length() >= 3);
    if (boost::starts_with(s, "NT(")) {
      assert (s[s.length() - 1] == ')');
      string subtype = s.substr(3, s.length() - 4);
      WordId sub_id = nt_vocab.Convert(subtype);

      assert (w2a.size() == (unsigned)i);
      Action a = {Action::kNT, sub_id};
      w2a.push_back(a);
      a2w[a] = i;
    }
    else if (boost::starts_with(s, "SHIFT(")) {
      assert (s[s.length() - 1] == ')');
      string subtype = s.substr(6, s.length() - 7);
      WordId sub_id = term_vocab.Convert(subtype);

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
  most_recent_source_thing = zeroes(cg, {hidden_dim});
}

void RnngOutputModel::SetDropout(float rate) {
  builder->SetDropout(rate);
}

Expression RnngOutputModel::GetState() const {
  return builder->GetStateVector(most_recent_source_thing);
}

Expression RnngOutputModel::GetState(RNNPointer p) const {
  assert (false);
}

RNNPointer RnngOutputModel::GetStatePointer() const {
  assert (false);
}

Expression RnngOutputModel::AddInput(const Word* prev_word, const Expression& context) {
  most_recent_source_thing = context;
  return AddInput(prev_word, context, builder->state());
}

Expression RnngOutputModel::AddInput(const Word* prev_word_, const Expression& context, const RNNPointer& p) {
  const StandardWord* prev_word = dynamic_cast<const StandardWord*>(prev_word_);
  most_recent_source_thing = context;
  Action action = Convert(prev_word->id);
  builder->PerformAction(action, p);
  return builder->GetStateVector(context);
}

Expression RnngOutputModel::PredictLogDistribution(const Expression& source_context) const {
  assert (false);
}

Word* RnngOutputModel::Sample(const Expression& source_context) const {
  assert (false);
}

Expression RnngOutputModel::Loss(const Expression& source_context, const Word* const ref) const {
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

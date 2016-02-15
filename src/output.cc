#include "output.h"
BOOST_CLASS_EXPORT_IMPLEMENT(SoftmaxOutputModel)
BOOST_CLASS_EXPORT_IMPLEMENT(MlpSoftmaxOutputModel)

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

Expression SoftmaxOutputModel::AddInput(const WordId prev_word, const Expression& context) {
  return AddInput(prev_word, context, output_builder.state());
}

Expression SoftmaxOutputModel::AddInput(const WordId prev_word, const Expression& context, const RNNPointer& p) {
  Expression prev_embedding = lookup(*pcg, embeddings, prev_word);
  Expression input = concatenate({prev_embedding, context});
  Expression state = output_builder.add_input(p, input);
  return state;
}

Expression SoftmaxOutputModel::PredictLogDistribution(const Expression& state) const {
  return fsb->full_log_distribution(state);
}

Expression SoftmaxOutputModel::Loss(const Expression& state, unsigned ref) const {
  return fsb->neg_log_softmax(state, ref);
}

WordId SoftmaxOutputModel::Sample(const Expression& state) const {
  return fsb->sample(state);
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

Expression MlpSoftmaxOutputModel::Loss(const Expression& state, unsigned ref) const {
  Expression h = tanh(affine_transform({b, W, state}));
  return SoftmaxOutputModel::Loss(h, ref);
}

WordId MlpSoftmaxOutputModel::Sample(const Expression& state) const {
  Expression h = tanh(affine_transform({b, W, state}));
  return SoftmaxOutputModel::Sample(h);
}

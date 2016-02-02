#include "output.h"
BOOST_CLASS_EXPORT_IMPLEMENT(SoftmaxOutputModel)

const unsigned lstm_layer_count = 2;

OutputModel::~OutputModel() {}

SoftmaxOutputModel::SoftmaxOutputModel() {}

SoftmaxOutputModel::SoftmaxOutputModel(Model& model, unsigned embedding_dim, unsigned context_dim, unsigned state_dim, Dict* vocab, const string& clusters_filename) : embedding_dim(embedding_dim), context_dim(context_dim), state_dim(state_dim), vocab_size(vocab->size()) {
  if (clusters_filename.length() > 0) {
    fsb = new ClassFactoredSoftmaxBuilder(state_dim, clusters_filename, vocab, &model);
  }
  else {
    fsb = new StandardSoftmaxBuilder(state_dim, vocab->size(), &model);
  }
  InitializeParameters(model);
}

void SoftmaxOutputModel::InitializeParameters(Model& model) {
  cerr << "Initializing SoftmaxOutputModel" << endl;
  embeddings = model.add_lookup_parameters(vocab_size, {embedding_dim});
  output_builder = LSTMBuilder(lstm_layer_count, embedding_dim + context_dim, state_dim, &model);
}

void SoftmaxOutputModel::NewGraph(ComputationGraph& cg) {
  output_builder.new_graph(cg);
  output_builder.start_new_sequence();
  fsb->new_graph(cg);
  pcg = &cg;
}

Expression SoftmaxOutputModel::GetState() {
  if (output_builder.h0.size() > 0) {
    return output_builder.back();
  }
  else {
    return zeroes(*pcg, {state_dim});
  }
}

Expression SoftmaxOutputModel::Loss(const WordId prev_word, const Expression& context, unsigned ref) {
  Expression prev_embedding = lookup(*pcg, embeddings, prev_word);
  Expression input = concatenate({prev_embedding, context});
  Expression state = output_builder.add_input(input);
  return fsb->neg_log_softmax(state, ref);
}

#include "encoder.h"
BOOST_CLASS_EXPORT_IMPLEMENT(BidirectionalSentenceEncoder)

const unsigned lstm_layer_count = 2;

BidirectionalSentenceEncoder::BidirectionalSentenceEncoder() {}

BidirectionalSentenceEncoder::BidirectionalSentenceEncoder(Model& model, unsigned vocab_size, unsigned input_dim, unsigned output_dim) : vocab_size(vocab_size), input_dim(input_dim), output_dim(output_dim) {
  assert (output_dim % 2 == 0);
  embeddings = model.add_lookup_parameters(vocab_size, {input_dim});
  forward_builder = LSTMBuilder(lstm_layer_count, input_dim, output_dim / 2, &model);
  reverse_builder = LSTMBuilder(lstm_layer_count, input_dim, output_dim / 2, &model);
}

void BidirectionalSentenceEncoder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
}

vector<Expression> BidirectionalSentenceEncoder::Encode(const TranslatorInput* input) {
  const Sentence& sentence = *dynamic_cast<const Sentence*>(input);
  vector<Expression> forward_encodings = EncodeForward(sentence);
  vector<Expression> reverse_encodings = EncodeReverse(sentence);
  vector<Expression> bidir_encodings(sentence.size());
  for (unsigned i = 0; i < sentence.size(); ++i) {
    const Expression& f = forward_encodings[i];
    const Expression& r = reverse_encodings[i];
    bidir_encodings[i] = concatenate({f, r});
  }
  return bidir_encodings;
}

vector<Expression> BidirectionalSentenceEncoder::EncodeForward(const Sentence& sentence) {
  forward_builder.new_graph(*pcg);
  forward_builder.start_new_sequence();
  vector<Expression> forward_encodings(sentence.size());
  for (unsigned i = 0; i < sentence.size(); ++i) {
    Expression x = lookup(*pcg, embeddings, sentence[i]);
    Expression y = forward_builder.add_input(x);
    forward_encodings[i] = y;
  }
  return forward_encodings;
}

vector<Expression> BidirectionalSentenceEncoder::EncodeReverse(const Sentence& sentence) {
  reverse_builder.new_graph(*pcg);
  reverse_builder.start_new_sequence();
  vector<Expression> reverse_encodings(sentence.size());
  for (unsigned i = 0; i < sentence.size(); ++i) {
    Expression x = lookup(*pcg, embeddings, sentence[i]);
    Expression y = reverse_builder.add_input(x);
    reverse_encodings[i] = y;
  }
  return reverse_encodings;
}

#include "encoder.h"
BOOST_CLASS_EXPORT_IMPLEMENT(BidirectionalSentenceEncoder)
BOOST_CLASS_EXPORT_IMPLEMENT(TrivialEncoder)

const unsigned lstm_layer_count = 2;

TrivialEncoder::TrivialEncoder() {}

TrivialEncoder::TrivialEncoder(Model& model, unsigned vocab_size, unsigned input_dim, unsigned output_dim) {
  embeddings = model.add_lookup_parameters(vocab_size, {input_dim});
  p_W = model.add_parameters({output_dim, input_dim});
  p_b = model.add_parameters({output_dim});
}

bool TrivialEncoder::IsT2S() const {
  return false;
}

void TrivialEncoder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
  W = parameter(cg, p_W);
  b = parameter(cg, p_b);
}

Expression TrivialEncoder::Embed(const Word* const word) {
  const StandardWord* standard_word = dynamic_cast<const StandardWord*>(word);
  assert (standard_word != nullptr);
  return lookup(*pcg, embeddings, standard_word->id);
}

vector<Expression> TrivialEncoder::Encode(const InputSentence* const input) {
  const LinearSentence& sentence = *dynamic_cast<const LinearSentence*>(input);
  vector<Expression> encodings(sentence.size());
  for (unsigned i = 0; i < sentence.size(); ++i) {
    Expression embedding = Embed(sentence[i]);
    encodings[i] = affine_transform({b, W, embedding});
  }
  return encodings;
}

BidirectionalSentenceEncoder::BidirectionalSentenceEncoder() {}

BidirectionalSentenceEncoder::BidirectionalSentenceEncoder(Model& model, unsigned vocab_size, unsigned input_dim, unsigned output_dim) {
  assert (output_dim % 2 == 0);
  embeddings = model.add_lookup_parameters(vocab_size, {input_dim});
  forward_builder = LSTMBuilder(lstm_layer_count, input_dim, output_dim / 2, &model);
  reverse_builder = LSTMBuilder(lstm_layer_count, input_dim, output_dim / 2, &model);
}

bool BidirectionalSentenceEncoder::IsT2S() const {
  return false;
}

void BidirectionalSentenceEncoder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
}

void BidirectionalSentenceEncoder::SetDropout(float rate) {
  forward_builder.set_dropout(rate);
  reverse_builder.set_dropout(rate);
}

vector<Expression> BidirectionalSentenceEncoder::Encode(const InputSentence* const input) {
  const LinearSentence& sentence = *dynamic_cast<const LinearSentence*>(input);
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

Expression BidirectionalSentenceEncoder::Embed(const Word* const word) {
  const StandardWord* standard_word = dynamic_cast<const StandardWord*>(word);
  assert (standard_word != nullptr);
  return lookup(*pcg, embeddings, standard_word->id);
}

vector<Expression> BidirectionalSentenceEncoder::EncodeForward(const LinearSentence& sentence) {
  forward_builder.new_graph(*pcg);
  forward_builder.start_new_sequence();
  vector<Expression> forward_encodings(sentence.size());
  for (unsigned i = 0; i < sentence.size(); ++i) {
    Expression x = Embed(sentence[i]);
    Expression y = forward_builder.add_input(x);
    forward_encodings[i] = y;
  }
  return forward_encodings;
}

vector<Expression> BidirectionalSentenceEncoder::EncodeReverse(const LinearSentence& sentence) {
  reverse_builder.new_graph(*pcg);
  reverse_builder.start_new_sequence();
  vector<Expression> reverse_encodings(sentence.size());
  for (unsigned i = 0; i < sentence.size(); ++i) {
    Expression x = Embed(sentence[i]);
    Expression y = reverse_builder.add_input(x);
    reverse_encodings[i] = y;
  }
  return reverse_encodings;
}

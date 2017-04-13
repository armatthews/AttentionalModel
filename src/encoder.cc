#include "encoder.h"
BOOST_CLASS_EXPORT_IMPLEMENT(BidirectionalEncoder)
BOOST_CLASS_EXPORT_IMPLEMENT(TrivialEncoder)

const unsigned lstm_layer_count = 2;

TrivialEncoder::TrivialEncoder() {}

TrivialEncoder::TrivialEncoder(Model& model, Embedder* embedder, unsigned output_dim) : embedder(embedder) {
  unsigned input_dim = embedder->Dim();
  p_W = model.add_parameters({output_dim, input_dim});
  p_b = model.add_parameters({output_dim});
}

void TrivialEncoder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
  embedder->NewGraph(cg);
  W = parameter(cg, p_W);
  b = parameter(cg, p_b);
}

vector<Expression> TrivialEncoder::Encode(const InputSentence* const input) {
  const LinearSentence& sentence = *dynamic_cast<const LinearSentence*>(input);
  vector<Expression> encodings(sentence.size());
  for (unsigned i = 0; i < sentence.size(); ++i) {
    Expression embedding = embedder->Embed(sentence[i]);
    encodings[i] = affine_transform({b, W, embedding});
  }
  return encodings;
}

Expression TrivialEncoder::EncodeSentence(const InputSentence* const input) {
  return sum(Encode(input));
}

BidirectionalEncoder::BidirectionalEncoder() {}

BidirectionalEncoder::BidirectionalEncoder(Model& model, Embedder* embedder, unsigned output_dim, bool peep_concat, bool peep_add) :
    embedder(embedder), output_dim(output_dim), peep_concat(peep_concat), peep_add(peep_add) {
  assert (output_dim % 2 == 0);
  unsigned input_dim = embedder->Dim();
  forward_builder = LSTMBuilder(lstm_layer_count, input_dim, output_dim / 2, model);
  reverse_builder = LSTMBuilder(lstm_layer_count, input_dim, output_dim / 2, model);
  forward_lstm_init = model.add_parameters({lstm_layer_count * output_dim / 2});
  reverse_lstm_init = model.add_parameters({lstm_layer_count * output_dim / 2});


  if (peep_add) {
    p_W = model.add_parameters({output_dim, input_dim});
  }
}

void BidirectionalEncoder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
  embedder->NewGraph(cg);

  forward_builder.new_graph(cg);
  reverse_builder.new_graph(cg);

  Expression forward_lstm_init_expr = parameter(cg, forward_lstm_init);
  forward_lstm_init_v = MakeLSTMInitialState(forward_lstm_init_expr, output_dim / 2, forward_builder.layers);

  Expression reverse_lstm_init_expr = parameter(cg, reverse_lstm_init);
  reverse_lstm_init_v = MakeLSTMInitialState(reverse_lstm_init_expr, output_dim / 2, reverse_builder.layers);

  if (peep_add) {
    W = parameter(*pcg, p_W);
  }
}

void BidirectionalEncoder::SetDropout(float rate) {
  forward_builder.set_dropout(rate);
  reverse_builder.set_dropout(rate);
}

vector<Expression> BidirectionalEncoder::Embed(const InputSentence* const input) {
  const LinearSentence& sentence = *dynamic_cast<const LinearSentence*>(input);
  vector<Expression> embeddings(sentence.size());
  for (unsigned i = 0; i < sentence.size(); ++i) {
    embeddings[i] = embedder->Embed(sentence[i]);
  }
  return embeddings;
}

vector<Expression> BidirectionalEncoder::Encode(const vector<Expression>& embeddings) {
  vector<Expression> forward_encodings = EncodeForward(embeddings);
  vector<Expression> reverse_encodings = EncodeReverse(embeddings);
  vector<Expression> bidir_encodings(embeddings.size());
  for (unsigned i = 0; i < embeddings.size(); ++i) {
    const Expression& f = forward_encodings[i];
    const Expression& r = reverse_encodings[embeddings.size() - 1 - i];
    bidir_encodings[i] = concatenate({f, r});
    if (peep_add) {
      bidir_encodings[i] = bidir_encodings[i] + W * embeddings[i];
    }
    if (peep_concat) {
      bidir_encodings[i] = concatenate({bidir_encodings[i], embeddings[i]});
    }
  }
  return bidir_encodings;
}

vector<Expression> BidirectionalEncoder::Encode(const InputSentence* const input) {
  return Encode(Embed(input));
}

vector<Expression> BidirectionalEncoder::EncodeForward(const vector<Expression>& embeddings) {
  forward_builder.start_new_sequence(forward_lstm_init_v);
  vector<Expression> forward_encodings(embeddings.size());
  for (unsigned i = 0; i < embeddings.size(); ++i) {
    Expression x = embeddings[i];
    Expression y = forward_builder.add_input(x);
    forward_encodings[i] = y;
  }
  return forward_encodings;
}

vector<Expression> BidirectionalEncoder::EncodeReverse(const vector<Expression>& embeddings) {
  reverse_builder.start_new_sequence(reverse_lstm_init_v);
  vector<Expression> reverse_encodings(embeddings.size());
  for (unsigned i = 0; i < embeddings.size(); ++i) {
    Expression x = embeddings[embeddings.size() - 1 - i];
    Expression y = reverse_builder.add_input(x);
    reverse_encodings[i] = y;
  }
  return reverse_encodings;
}

Expression BidirectionalEncoder::EncodeSentence(const InputSentence* const input) {
  const LinearSentence& sentence = *dynamic_cast<const LinearSentence*>(input);
  vector<Expression> embeddings(sentence.size());
  for (unsigned i = 0; i < sentence.size(); ++i) {
    embeddings[i] = embedder->Embed(sentence[i]);
  }
  vector<Expression> forward_encodings = EncodeForward(embeddings);
  vector<Expression> reverse_encodings = EncodeReverse(embeddings);
  return concatenate({forward_encodings[0], reverse_encodings[sentence.size() - 1]});
}

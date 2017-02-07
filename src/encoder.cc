#include "encoder.h"
BOOST_CLASS_EXPORT_IMPLEMENT(BidirectionalSentenceEncoder)
BOOST_CLASS_EXPORT_IMPLEMENT(TrivialEncoder)
BOOST_CLASS_EXPORT_IMPLEMENT(MorphologyEncoder)

const unsigned lstm_layer_count = 2;

TrivialEncoder::TrivialEncoder() {}

TrivialEncoder::TrivialEncoder(Model& model, unsigned vocab_size, unsigned input_dim, unsigned output_dim) {
  embeddings = model.add_lookup_parameters(vocab_size, {input_dim});
  p_W = model.add_parameters({output_dim, input_dim});
  p_b = model.add_parameters({output_dim});
}

void TrivialEncoder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
  W = parameter(cg, p_W);
  b = parameter(cg, p_b);
}

Expression TrivialEncoder::Embed(const shared_ptr<const Word> word) {
  const shared_ptr<const StandardWord> standard_word = dynamic_pointer_cast<const StandardWord>(word);
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

BidirectionalSentenceEncoder::BidirectionalSentenceEncoder(Model& model, unsigned vocab_size, unsigned input_dim, unsigned output_dim, bool peep_concat, bool peep_add) :
    output_dim(output_dim), peep_concat(peep_concat), peep_add(peep_add) {
  assert (output_dim % 2 == 0);
  embeddings = model.add_lookup_parameters(vocab_size, {input_dim});
  forward_builder = LSTMBuilder(lstm_layer_count, input_dim, output_dim / 2, model);
  reverse_builder = LSTMBuilder(lstm_layer_count, input_dim, output_dim / 2, model);
  forward_lstm_init = model.add_parameters({lstm_layer_count * output_dim / 2});
  reverse_lstm_init = model.add_parameters({lstm_layer_count * output_dim / 2});
}

void BidirectionalSentenceEncoder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
  forward_builder.new_graph(cg);
  reverse_builder.new_graph(cg);

  Expression forward_lstm_init_expr = parameter(cg, forward_lstm_init);
  forward_lstm_init_v = MakeLSTMInitialState(forward_lstm_init_expr, output_dim / 2, forward_builder.layers);

  Expression reverse_lstm_init_expr = parameter(cg, reverse_lstm_init);
  reverse_lstm_init_v = MakeLSTMInitialState(reverse_lstm_init_expr, output_dim / 2, reverse_builder.layers);
}

void BidirectionalSentenceEncoder::SetDropout(float rate) {
  forward_builder.set_dropout(rate);
  reverse_builder.set_dropout(rate);
}

vector<Expression> BidirectionalSentenceEncoder::Encode(const InputSentence* const input) {
  const LinearSentence& sentence = *dynamic_cast<const LinearSentence*>(input);
  vector<Expression> embeddings(sentence.size());
  for (unsigned i = 0; i < sentence.size(); ++i) {
    embeddings[i] = Embed(sentence[i]);
  }
  vector<Expression> forward_encodings = EncodeForward(embeddings);
  vector<Expression> reverse_encodings = EncodeReverse(embeddings);
  vector<Expression> bidir_encodings(sentence.size());
  for (unsigned i = 0; i < sentence.size(); ++i) {
    const Expression& f = forward_encodings[i];
    const Expression& r = reverse_encodings[sentence.size() - 1 - i];
    bidir_encodings[i] = concatenate({f, r});
    if (peep_add) {
      // XXX: If the word embedding dim is not the same as the LSTM dim there needs to be a transformation matrix here
      bidir_encodings[i] = bidir_encodings[i] + embeddings[i];
    }
    if (peep_concat) {
      bidir_encodings[i] = concatenate({bidir_encodings[i], embeddings[i]});
    }
  }
  return bidir_encodings;
}

Expression BidirectionalSentenceEncoder::Embed(const shared_ptr<const Word> word) {
  const shared_ptr<const StandardWord> standard_word = dynamic_pointer_cast<const StandardWord>(word);
  assert (standard_word != nullptr);
  return lookup(*pcg, embeddings, standard_word->id);
}

vector<Expression> BidirectionalSentenceEncoder::EncodeForward(const vector<Expression>& embeddings) {
  forward_builder.start_new_sequence(forward_lstm_init_v);
  vector<Expression> forward_encodings(embeddings.size());
  for (unsigned i = 0; i < embeddings.size(); ++i) {
    Expression x = embeddings[i];
    Expression y = forward_builder.add_input(x);
    forward_encodings[i] = y;
  }
  return forward_encodings;
}

vector<Expression> BidirectionalSentenceEncoder::EncodeReverse(const vector<Expression>& embeddings) {
  reverse_builder.start_new_sequence(reverse_lstm_init_v);
  vector<Expression> reverse_encodings(embeddings.size());
  for (unsigned i = 0; i < embeddings.size(); ++i) {
    Expression x = embeddings[embeddings.size() - 1 - i];
    Expression y = reverse_builder.add_input(x);
    reverse_encodings[i] = y;
  }
  return reverse_encodings;
}

MorphologyEncoder::MorphologyEncoder() {}

MorphologyEncoder::MorphologyEncoder(Model& model, unsigned word_vocab_size, unsigned root_vocab_size, unsigned affix_vocab_size, unsigned char_vocab_size, unsigned word_emb_dim, unsigned affix_emb_dim, unsigned char_emb_dim,
    unsigned affix_lstm_dim, unsigned char_lstm_dim, unsigned main_lstm_dim, bool peep_concat, bool peep_add)
  : main_lstm_dim(main_lstm_dim), peep_concat(peep_concat), peep_add(peep_add), embedder(MorphologyEmbedder(model, word_vocab_size, root_vocab_size, affix_vocab_size, char_vocab_size, word_emb_dim, affix_emb_dim, char_emb_dim, affix_lstm_dim, char_lstm_dim)) {
  assert (main_lstm_dim % 2 == 0);
  unsigned total_dim = word_emb_dim + affix_lstm_dim + char_lstm_dim;
  forward_builder = LSTMBuilder(lstm_layer_count, total_dim, main_lstm_dim / 2, model);
  reverse_builder = LSTMBuilder(lstm_layer_count, total_dim, main_lstm_dim / 2, model);

  forward_lstm_init = model.add_parameters({lstm_layer_count * main_lstm_dim / 2});
  reverse_lstm_init = model.add_parameters({lstm_layer_count * main_lstm_dim / 2});
}

void MorphologyEncoder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
  forward_builder.new_graph(cg);
  reverse_builder.new_graph(cg);
  embedder.NewGraph(cg);

  Expression forward_lstm_init_expr = parameter(cg, forward_lstm_init);
  forward_lstm_init_v = MakeLSTMInitialState(forward_lstm_init_expr, main_lstm_dim / 2, forward_builder.layers);

  Expression reverse_lstm_init_expr = parameter(cg, reverse_lstm_init);
  reverse_lstm_init_v = MakeLSTMInitialState(reverse_lstm_init_expr, main_lstm_dim / 2, reverse_builder.layers);
}

void MorphologyEncoder::SetDropout(float rate) {}

vector<Expression> MorphologyEncoder::Encode(const InputSentence* const input) {
  const LinearSentence& sentence = *dynamic_cast<const LinearSentence*>(input);
  vector<Expression> embeddings(sentence.size());
  for (unsigned i = 0; i < sentence.size(); ++i) {
    embeddings[i] = embedder.Embed(sentence[i]);
  }
  vector<Expression> forward_encodings = EncodeForward(embeddings);
  vector<Expression> reverse_encodings = EncodeReverse(embeddings);
  assert (forward_encodings.size() == reverse_encodings.size());

  vector<Expression> encodings(forward_encodings.size());
  for (unsigned i = 0; i < forward_encodings.size(); ++i) {
    encodings[i] = concatenate({forward_encodings[i], reverse_encodings[i]});
    if (peep_add) {
      // XXX: If the word embedding dim is not the same as the LSTM dim there needs to be a transformation matrix here
      encodings[i] = encodings[i] + embeddings[i];
    }
    if (peep_concat) {
     encodings[i] = concatenate({encodings[i], embeddings[i]});
    }
  }

  return encodings;
}

vector<Expression> MorphologyEncoder::EncodeForward(const vector<Expression>& embeddings) {
  forward_builder.start_new_sequence(forward_lstm_init_v);

  vector<Expression> r;
  for (Expression emb : embeddings) {
    Expression e = forward_builder.add_input(emb);
    r.push_back(e);
  }

  return r;
}

vector<Expression> MorphologyEncoder::EncodeReverse(const vector<Expression>& embeddings) {
  reverse_builder.start_new_sequence(reverse_lstm_init_v);

  vector<Expression> r;
  for (auto it = embeddings.rbegin(); it != embeddings.rend(); ++it) {
    Expression emb = *it;
    Expression e = reverse_builder.add_input(emb);
    r.push_back(e);
  }

  return r;
}

#include "morphology.h"

const unsigned lstm_layer_count = 2;

MorphologyEmbedder::MorphologyEmbedder() {}

MorphologyEmbedder::MorphologyEmbedder(Model& model, unsigned word_vocab_size, unsigned root_vocab_size, unsigned affix_vocab_size, unsigned char_vocab_size, unsigned word_emb_dim, unsigned affix_emb_dim, unsigned char_emb_dim, unsigned affix_lstm_dim, unsigned char_lstm_dim) : affix_lstm_dim(affix_lstm_dim), char_lstm_dim(char_lstm_dim) {
  word_embeddings = model.add_lookup_parameters(word_vocab_size, {word_emb_dim});
  root_embeddings = model.add_lookup_parameters(root_vocab_size, {2 * affix_lstm_dim});
  affix_embeddings = model.add_lookup_parameters(affix_vocab_size, {affix_emb_dim});
  char_embeddings = model.add_lookup_parameters(char_vocab_size, {char_emb_dim});

  morph_lstm = LSTMBuilder(lstm_layer_count, affix_emb_dim, affix_lstm_dim, &model);
  char_lstm = LSTMBuilder(lstm_layer_count, char_emb_dim, char_lstm_dim, &model);

  char_lstm_init = model.add_parameters({lstm_layer_count * char_lstm_dim});
}

void MorphologyEmbedder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
  char_lstm.new_graph(cg);
  morph_lstm.new_graph(cg);

  Expression char_lstm_init_expr = parameter(cg, char_lstm_init);
  char_lstm_init_v = MakeLSTMInitialState(char_lstm_init_expr, char_lstm_dim, char_lstm.layers);
}

void MorphologyEmbedder::SetDropout(float rate) {}

Expression MorphologyEmbedder::Embed(const Word* const word) {
  const MorphoWord* mword = dynamic_cast<const MorphoWord*>(word);

  Expression word_emb = lookup(*pcg, word_embeddings, mword->word);

  vector<Expression> analysis_embs;
  for (const Analysis analysis : mword->analyses) {
    Expression root_emb = lookup(*pcg, root_embeddings, analysis.root);
    vector<Expression> init = MakeLSTMInitialState(root_emb, affix_lstm_dim, morph_lstm.layers);
    morph_lstm.start_new_sequence(init);
    for (WordId affix : analysis.affixes) {
      Expression affix_emb = lookup(*pcg, affix_embeddings, affix);
      morph_lstm.add_input(affix_emb);
    }
    Expression analysis_emb = morph_lstm.back();
    analysis_embs.push_back(analysis_emb);
  }

  // Ghetto max pooling
  assert (analysis_embs.size() > 0);
  Expression morph_emb = analysis_embs[0];
  for (unsigned i = 1; i < analysis_embs.size(); ++i) {
    morph_emb = max(morph_emb, analysis_embs[i]);
  }

  char_lstm.start_new_sequence(char_lstm_init_v);
  for (WordId c : mword->chars) {
    Expression c_emb = lookup(*pcg, char_embeddings, c);
    char_lstm.add_input(c_emb);
  }
  Expression char_emb = char_lstm.back();

  return concatenate({word_emb, morph_emb, char_emb});
}


#include "embedder.h"
BOOST_CLASS_EXPORT_IMPLEMENT(StandardEmbedder)
BOOST_CLASS_EXPORT_IMPLEMENT(MorphologyEmbedder)

const unsigned lstm_layer_count = 2;

void Embedder::NewGraph(ComputationGraph& cg) {}
void Embedder::SetDropout(float) {}

StandardEmbedder::StandardEmbedder() {}

StandardEmbedder::StandardEmbedder(Model& model, unsigned vocab_size, unsigned emb_dim) : emb_dim(emb_dim), pcg(nullptr) {
  embeddings = model.add_lookup_parameters(vocab_size, {emb_dim});
}

void StandardEmbedder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
}

void StandardEmbedder::SetDropout(float) {}

unsigned StandardEmbedder::Dim() const {
  return emb_dim;
}

Expression StandardEmbedder::Embed(const shared_ptr<const Word> word) {
  const shared_ptr<const StandardWord> standard_word = dynamic_pointer_cast<const StandardWord>(word);
  assert (standard_word != nullptr);
  return lookup(*pcg, embeddings, standard_word->id);
}

MorphologyEmbedder::MorphologyEmbedder() {}

MorphologyEmbedder::MorphologyEmbedder(Model& model, unsigned word_vocab_size, unsigned root_vocab_size, unsigned affix_vocab_size, unsigned char_vocab_size, unsigned word_emb_dim, unsigned affix_emb_dim, unsigned char_emb_dim, unsigned affix_lstm_dim, unsigned char_lstm_dim, bool use_words, bool use_morphology) : use_words(use_words), use_morphology(use_morphology), affix_lstm_dim(affix_lstm_dim), char_lstm_dim(char_lstm_dim) {
  total_emb_dim = 0;

  if (use_words) {
    word_embeddings = model.add_lookup_parameters(word_vocab_size, {word_emb_dim});
    total_emb_dim += word_emb_dim;
  }

  if (use_morphology) {
    root_embeddings = model.add_lookup_parameters(root_vocab_size, {2 * affix_lstm_dim});
    affix_embeddings = model.add_lookup_parameters(affix_vocab_size, {affix_emb_dim});
    morph_lstm = LSTMBuilder(lstm_layer_count, affix_emb_dim, affix_lstm_dim, model);
    total_emb_dim += affix_lstm_dim;
  }

  char_embeddings = model.add_lookup_parameters(char_vocab_size, {char_emb_dim});
  char_lstm = LSTMBuilder(lstm_layer_count, char_emb_dim, char_lstm_dim, model);
  char_lstm_init = model.add_parameters({lstm_layer_count * char_lstm_dim});
  total_emb_dim += char_lstm_dim;
}

void MorphologyEmbedder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
  char_lstm.new_graph(cg);
  if (use_morphology) {
    morph_lstm.new_graph(cg);
  }

  Expression char_lstm_init_expr = parameter(cg, char_lstm_init);
  char_lstm_init_v = MakeLSTMInitialState(char_lstm_init_expr, char_lstm_dim, char_lstm.layers);
}

void MorphologyEmbedder::SetDropout(float rate) {}

unsigned MorphologyEmbedder::Dim() const {
  return total_emb_dim;
}

Expression MorphologyEmbedder::EmbedWord(WordId word) {
  return lookup(*pcg, word_embeddings, word);
}

Expression MorphologyEmbedder::EmbedAnalysis(const Analysis& analysis) {
    Expression root_emb = lookup(*pcg, root_embeddings, analysis.root);
    vector<Expression> init = MakeLSTMInitialState(root_emb, affix_lstm_dim, morph_lstm.layers);
    morph_lstm.start_new_sequence(init);
    for (WordId affix : analysis.affixes) {
      Expression affix_emb = lookup(*pcg, affix_embeddings, affix);
      morph_lstm.add_input(affix_emb);
    }
    Expression analysis_emb = morph_lstm.back();
}

Expression MorphologyEmbedder::PoolAnalysisEmbeddings(const vector<Expression> analysis_embs) {
  // Ghetto max pooling
  assert (analysis_embs.size() > 0);
  Expression morph_emb = analysis_embs[0];
  for (unsigned i = 1; i < analysis_embs.size(); ++i) {
    morph_emb = max(morph_emb, analysis_embs[i]);
  }
  return morph_emb;
}

Expression MorphologyEmbedder::EmbedAnalyses(const vector<Analysis>& analyses) {
  vector<Expression> analysis_embs;
  for (const Analysis analysis : analyses) {
    Expression analysis_emb = EmbedAnalysis(analysis);
    analysis_embs.push_back(analysis_emb);
  }
  return PoolAnalysisEmbeddings(analysis_embs);
}

Expression MorphologyEmbedder::EmbedCharSequence(const vector<WordId>& chars) {
  char_lstm.start_new_sequence(char_lstm_init_v);
  for (const WordId c : chars) {
    Expression c_emb = lookup(*pcg, char_embeddings, c);
    char_lstm.add_input(c_emb);
  }
  Expression char_emb = char_lstm.back();
  return char_emb;
}

Expression MorphologyEmbedder::Embed(const shared_ptr<const Word> word) {
  const shared_ptr<const MorphoWord> mword = dynamic_pointer_cast<const MorphoWord>(word);

  vector<Expression> pieces;
  if (use_words) {
    Expression word_emb = EmbedWord(mword->word);
    pieces.push_back(word_emb);
  }

  if (use_morphology) {
    Expression morph_emb = EmbedAnalyses(mword->analyses);
    pieces.push_back(morph_emb);
  }

  Expression char_emb = EmbedCharSequence(mword->chars);
  pieces.push_back(char_emb);

  return concatenate(pieces);
}


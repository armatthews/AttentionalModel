#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include "dynet/dynet.h"
#include "dynet/lstm.h"
#include "dynet/expr.h"
#include "utils.h"

class MorphologyEmbedder {
public:
  MorphologyEmbedder();
  // Note: We don't need root_emb_dim since root_emb_dim = 2 * affix_lstm_dim.
  MorphologyEmbedder(Model& model, unsigned word_vocab_size, unsigned root_vocab_size, unsigned affix_vocab_size, unsigned char_vocab_size, unsigned word_emb_dim, unsigned affix_emb_dim, unsigned char_emb_dim, unsigned affix_lstm_dim, unsigned char_lstm_dim);
  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  Expression Embed(const Word* const word);
private:
  unsigned affix_lstm_dim, char_lstm_dim;
  LookupParameter word_embeddings;
  LookupParameter root_embeddings;
  LookupParameter affix_embeddings;
  LookupParameter char_embeddings;
  LSTMBuilder char_lstm;
  LSTMBuilder morph_lstm;
  // This is the initial state for the char LSTM.
  // Note that the affix LSTM is initialized with
  // the embedding for the root, so there's no need
  // for an additional parameter for that.
  Parameter char_lstm_init;
  vector<Expression> char_lstm_init_v;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & affix_lstm_dim & char_lstm_dim;
    ar & word_embeddings;
    ar & root_embeddings;
    ar & affix_embeddings;
    ar & char_embeddings;
    ar & char_lstm;
    ar & morph_lstm;
    ar & char_lstm_init;
    //Parameter& prev = char_lstm.params.back().back();
    //char_lstm_init = Parameter(prev.mp, prev.index + 1); // XXX: Super duper hacky
  }
};

#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "dynet/dynet.h"
#include "dynet/lstm.h"
#include "dynet/expr.h"
#include "utils.h"

class Embedder {
public:
  virtual void NewGraph(ComputationGraph& cg);
  virtual void SetDropout(float rate);
  virtual unsigned Dim() const = 0;
  virtual Expression Embed(const shared_ptr<const Word> word) = 0;
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class StandardEmbedder : public Embedder {
public:
  StandardEmbedder();
  StandardEmbedder(Model& model, unsigned vocab_size, unsigned emb_dim);

  void NewGraph(ComputationGraph& cg) override;
  void SetDropout(float rate) override;
  unsigned Dim() const override;
  Expression Embed(const shared_ptr<const Word> word) override;
private:
  unsigned emb_dim;
  LookupParameter embeddings;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<Embedder>(*this);
    ar & emb_dim;
    ar & embeddings;
  }
};
BOOST_CLASS_EXPORT_KEY(StandardEmbedder)

class MorphologyEmbedder : public Embedder {
public:
  MorphologyEmbedder();
  // Note: We don't need root_emb_dim since root_emb_dim = 2 * affix_lstm_dim.
  MorphologyEmbedder(Model& model, unsigned word_vocab_size, unsigned root_vocab_size, unsigned affix_vocab_size, unsigned char_vocab_size, unsigned word_emb_dim, unsigned affix_emb_dim, unsigned char_emb_dim, unsigned affix_lstm_dim, unsigned char_lstm_dim, bool use_words, bool use_morphology);
  void NewGraph(ComputationGraph& cg) override;
  void SetDropout(float rate) override;
  unsigned Dim() const;
  Expression EmbedWord(WordId word);
  Expression EmbedAnalysis(const Analysis& analysis);
  Expression PoolAnalysisEmbeddings(const vector<Expression> analysis_embs);
  Expression EmbedAnalyses(const vector<Analysis>& analyses);
  Expression EmbedCharSequence(const vector<WordId>& chars);
  Expression Embed(const shared_ptr<const Word> word) override;
private:
  bool use_words;
  bool use_morphology;
  unsigned total_emb_dim;
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
    ar & boost::serialization::base_object<Embedder>(*this);
    ar & use_words;
    ar & use_morphology;
    ar & total_emb_dim;
    ar & affix_lstm_dim & char_lstm_dim;
    ar & word_embeddings;
    ar & root_embeddings;
    ar & affix_embeddings;
    ar & char_embeddings;
    ar & char_lstm;
    ar & morph_lstm;
    ar & char_lstm_init;
  }
};
BOOST_CLASS_EXPORT_KEY(MorphologyEmbedder)

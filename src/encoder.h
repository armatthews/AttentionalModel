#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "cnn/expr.h"
#include "utils.h"
#include "morphology.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class EncoderModel {
public:
  virtual ~EncoderModel() {}

  virtual void NewGraph(ComputationGraph& cg) = 0;
  virtual void SetDropout(float rate) {};
  virtual vector<Expression> Encode(const InputSentence* const input) = 0;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class TrivialEncoder : public EncoderModel {
public:
  TrivialEncoder();
  TrivialEncoder(Model& model, unsigned vocab_size, unsigned input_dim, unsigned output_dim);

  void NewGraph(ComputationGraph& cg);
  vector<Expression> Encode(const InputSentence* const input);
  Expression Embed(const Word* const word);
private:
  Parameter p_W, p_b;
  Expression W, b;
  LookupParameter embeddings;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<EncoderModel>(*this);
    ar & p_W;
    ar & p_b;
    ar & embeddings;
  }
};
BOOST_CLASS_EXPORT_KEY(TrivialEncoder)

class BidirectionalSentenceEncoder : public EncoderModel {
public:
  BidirectionalSentenceEncoder();
  BidirectionalSentenceEncoder(Model& model, unsigned vocab_size, unsigned input_dim, unsigned output_dim);

  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  vector<Expression> Encode(const InputSentence* const input);
  vector<Expression> EncodeForward(const LinearSentence& sentence);
  vector<Expression> EncodeReverse(const LinearSentence& sentence);
  Expression Embed(const Word* const word);
private:
  LSTMBuilder forward_builder;
  LSTMBuilder reverse_builder;
  LookupParameter embeddings;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<EncoderModel>(*this);
    ar & forward_builder;
    ar & reverse_builder;
    ar & embeddings;
  }
};
BOOST_CLASS_EXPORT_KEY(BidirectionalSentenceEncoder)

class MorphologyEncoder : public EncoderModel {
public:
  MorphologyEncoder();
  MorphologyEncoder(Model& model, unsigned word_vocab_size, unsigned root_vocab_size, unsigned affix_vocab_size, unsigned char_vocab_size, unsigned word_emb_dim, unsigned affix_emb_dim, unsigned char_emb_dim,
    unsigned affix_lstm_dim, unsigned char_lstm_dim, unsigned main_lstm_dim);

  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  vector<Expression> Encode(const InputSentence* const input);
  vector<Expression> EncodeForward(const LinearSentence& sentence);
  vector<Expression> EncodeReverse(const LinearSentence& sentence);

private:
  unsigned main_lstm_dim;
  LSTMBuilder forward_builder;
  LSTMBuilder reverse_builder;
  Parameter forward_lstm_init;
  vector<Expression> forward_lstm_init_v;
  Parameter reverse_lstm_init;
  vector<Expression> reverse_lstm_init_v;
  MorphologyEmbedder embedder;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<EncoderModel>(*this);
    ar & main_lstm_dim;
    ar & forward_builder;
    ar & reverse_builder;
    ar & forward_lstm_init;
    ar & reverse_lstm_init;
    ar & embedder;
  }
};
BOOST_CLASS_EXPORT_KEY(MorphologyEncoder)


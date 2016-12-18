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
#include "morphology.h"

using namespace std;
using namespace dynet;
using namespace dynet::expr;

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
  Expression Embed(const shared_ptr<const Word> word);
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
  BidirectionalSentenceEncoder(Model& model, unsigned vocab_size, unsigned input_dim, unsigned output_dim, bool peep_concat, bool peep_add);

  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  vector<Expression> Encode(const InputSentence* const input);
  vector<Expression> EncodeForward(const vector<Expression>& embeddings);
  vector<Expression> EncodeReverse(const vector<Expression>& embeddings);
  Expression Embed(const shared_ptr<const Word> word);
private:
  unsigned output_dim;
  bool peep_concat, peep_add;
  LSTMBuilder forward_builder;
  LSTMBuilder reverse_builder;
  Parameter forward_lstm_init;
  vector<Expression> forward_lstm_init_v;
  Parameter reverse_lstm_init;
  vector<Expression> reverse_lstm_init_v;
  LookupParameter embeddings;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<EncoderModel>(*this);
    ar & output_dim;
    ar & peep_concat & peep_add;
    ar & forward_builder;
    ar & reverse_builder;
    ar & forward_lstm_init;
    ar & reverse_lstm_init;
    ar & embeddings;
  }
};
BOOST_CLASS_EXPORT_KEY(BidirectionalSentenceEncoder)

class MorphologyEncoder : public EncoderModel {
public:
  MorphologyEncoder();
  MorphologyEncoder(Model& model, unsigned word_vocab_size, unsigned root_vocab_size, unsigned affix_vocab_size, unsigned char_vocab_size, unsigned word_emb_dim, unsigned affix_emb_dim, unsigned char_emb_dim,
    unsigned affix_lstm_dim, unsigned char_lstm_dim, unsigned main_lstm_dim, bool peep_concat, bool peep_add);

  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  vector<Expression> Encode(const InputSentence* const input);
  vector<Expression> EncodeForward(const vector<Expression>& embeddings);
  vector<Expression> EncodeReverse(const vector<Expression>& embeddings);

private:
  unsigned main_lstm_dim;
  bool peep_concat, peep_add;
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
    ar & peep_concat & peep_add;
    ar & forward_builder;
    ar & reverse_builder;
    ar & forward_lstm_init;
    ar & reverse_lstm_init;
    ar & embedder;
  }
};
BOOST_CLASS_EXPORT_KEY(MorphologyEncoder)


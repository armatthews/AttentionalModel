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
#include "embedder.h"

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
  TrivialEncoder(Model& model, Embedder* embedder, unsigned output_dim);

  void NewGraph(ComputationGraph& cg);
  vector<Expression> Encode(const InputSentence* const input);
private:
  Embedder* embedder;
  Parameter p_W, p_b;
  Expression W, b;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<EncoderModel>(*this);
    ar & embedder;
    ar & p_W;
    ar & p_b;
  }
};
BOOST_CLASS_EXPORT_KEY(TrivialEncoder)

class BidirectionalEncoder : public EncoderModel {
public:
  BidirectionalEncoder();
  BidirectionalEncoder(Model& model, Embedder* embedder, unsigned output_dim, bool peep_concat, bool peep_add);

  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  vector<Expression> Encode(const InputSentence* const input);
  vector<Expression> EncodeForward(const vector<Expression>& embeddings);
  vector<Expression> EncodeReverse(const vector<Expression>& embeddings);
private:
  Embedder* embedder;
  unsigned output_dim;
  bool peep_concat, peep_add;
  LSTMBuilder forward_builder;
  LSTMBuilder reverse_builder;
  Parameter forward_lstm_init;
  vector<Expression> forward_lstm_init_v;
  Parameter reverse_lstm_init;
  vector<Expression> reverse_lstm_init_v;
  Parameter p_W;
  Expression W;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<EncoderModel>(*this);
    ar & embedder;
    ar & output_dim;
    ar & peep_concat & peep_add;
    ar & forward_builder;
    ar & reverse_builder;
    ar & forward_lstm_init;
    ar & reverse_lstm_init;
    ar & p_W;
  }
};
BOOST_CLASS_EXPORT_KEY(BidirectionalEncoder)

class MultiFactorEncoder : public EncoderModel {
public:
  MultiFactorEncoder();
  MultiFactorEncoder(const vector<EncoderModel*>& encoders);
  void NewGraph(ComputationGraph& cg) override;
  void SetDropout(float rate) override;
  vector<Expression> Encode(const InputSentence* const input) override;
//private:
  vector<EncoderModel*> encoders;
};
BOOST_CLASS_EXPORT_KEY(MultiFactorEncoder)


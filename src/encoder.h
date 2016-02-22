#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "cnn/expr.h"
#include "utils.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class EncoderModel {
public:
  virtual ~EncoderModel() {}

  virtual bool IsT2S() const = 0;
  virtual void NewGraph(ComputationGraph& cg) = 0;
  virtual void SetDropout(float rate) {};
  virtual vector<Expression> Encode(const TranslatorInput* const input) = 0;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class TrivialEncoder : public EncoderModel {
public:
  TrivialEncoder();
  TrivialEncoder(Model& model, unsigned vocab_size, unsigned input_dim, unsigned output_dim);

  bool IsT2S() const;
  void NewGraph(ComputationGraph& cg);
  vector<Expression> Encode(const TranslatorInput* const input);
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

  bool IsT2S() const;
  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  vector<Expression> Encode(const TranslatorInput* const input);
  vector<Expression> EncodeForward(const Sentence& sentence);
  vector<Expression> EncodeReverse(const Sentence& sentence);
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

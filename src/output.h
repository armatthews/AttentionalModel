#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "cnn/expr.h"
#include "cnn/cfsm-builder.h"
#include "utils.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class OutputModel {
public:
  virtual ~OutputModel();

  virtual void NewGraph(ComputationGraph& cg) = 0;
  virtual void SetDropout(float rate) {}
  virtual Expression GetState() const = 0;
  virtual Expression GetState(RNNPointer p) const = 0;
  virtual RNNPointer GetStatePointer() const = 0;
  virtual Expression AddInput(const WordId prev_word, const Expression& context) = 0;
  virtual Expression AddInput(const WordId prev_word, const Expression& context, const RNNPointer& p) = 0;
  virtual Expression PredictLogDistribution(const Expression& state) const = 0;
  virtual WordId Sample(const Expression& state) const = 0;
  virtual Expression Loss(const Expression& state, unsigned ref) const = 0;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class SoftmaxOutputModel : public OutputModel {
public:
  SoftmaxOutputModel();
  SoftmaxOutputModel(Model& model, unsigned embedding_dim, unsigned context_dim, unsigned state_dim, Dict* vocab, const string& clusters_file);

  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  Expression GetState() const;
  Expression GetState(RNNPointer p) const;
  RNNPointer GetStatePointer() const;
  Expression AddInput(const WordId prev_word, const Expression& context);
  Expression AddInput(const WordId prev_word, const Expression& context, const RNNPointer& p);
  Expression PredictLogDistribution(const Expression& state) const;
  WordId Sample(const Expression& state) const;
  Expression Loss(const Expression& state, unsigned ref) const;
protected:
  unsigned state_dim;
  LSTMBuilder output_builder;
  LookupParameter embeddings;
  SoftmaxBuilder* fsb;
  ComputationGraph* pcg;
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<OutputModel>(*this);
    ar & state_dim;
    ar & output_builder;
    ar & embeddings;
    ar & fsb;
  }
};
BOOST_CLASS_EXPORT_KEY(SoftmaxOutputModel)

class MlpSoftmaxOutputModel : public SoftmaxOutputModel {
public:
  MlpSoftmaxOutputModel();
  MlpSoftmaxOutputModel(Model& model, unsigned embedding_dim, unsigned context_dim, unsigned state_dim, unsigned hidden_dim, Dict* vocab, const string& clusters_file);

  void NewGraph(ComputationGraph& cg);
  Expression PredictLogDistribution(const Expression& state) const;
  WordId Sample(const Expression& state) const;
  Expression Loss(const Expression& state, unsigned ref) const;
private:
  Parameter p_W, p_b;
  Expression W, b;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<SoftmaxOutputModel>(*this);
    ar & p_W;
    ar & p_b;
  }  
};
BOOST_CLASS_EXPORT_KEY(MlpSoftmaxOutputModel)

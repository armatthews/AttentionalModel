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

class AttentionModel {
public:
  virtual ~AttentionModel();

  virtual void NewGraph(ComputationGraph& cg) = 0;
  virtual void SetDropout(float rate) {}
  virtual Expression GetScoreVector(const vector<Expression>& inputs, const Expression& state) = 0;
  virtual Expression GetAlignmentVector(const vector<Expression>& inputs, const Expression& state) = 0;
  virtual Expression GetContext(const vector<Expression>& inputs, const Expression& state) = 0;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class StandardAttentionModel : public AttentionModel {
public:
  StandardAttentionModel();
  StandardAttentionModel(Model& model, unsigned input_dim, unsigned state_dim, unsigned hidden_dim);

  void NewGraph(ComputationGraph& cg);
  Expression GetScoreVector(const vector<Expression>& inputs, const Expression& state);
  Expression GetAlignmentVector(const vector<Expression>& inputs, const Expression& state);
  Expression GetContext(const vector<Expression>& inputs, const Expression& state);
private:
  Parameter p_U, p_V, p_W, p_b;
  Expression U, V, W, b;
  Expression WI, input_matrix;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    //boost::serialization::void_cast_register<StandardAttentionModel, AttentionModel>();
    ar & boost::serialization::base_object<AttentionModel>(*this);
    ar & p_U;
    ar & p_V;
    ar & p_W;
    ar & p_b;
  }
};
BOOST_CLASS_EXPORT_KEY(StandardAttentionModel)

class EncoderDecoderAttentionModel : public AttentionModel {
public:
  EncoderDecoderAttentionModel();
  EncoderDecoderAttentionModel(Model& model, unsigned input_dim, unsigned state_dim);

  void NewGraph(ComputationGraph& cg);
  Expression GetScoreVector(const vector<Expression>& inputs, const Expression& state);
  Expression GetAlignmentVector(const vector<Expression>& inputs, const Expression& state);
  Expression GetContext(const vector<Expression>& inputs, const Expression& state);
private:
  unsigned state_dim;
  Parameter p_W, p_b;
  Expression W, b;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    //boost::serialization::void_cast_register<EncoderDecoderAttentionModel, AttentionModel>();
    ar & boost::serialization::base_object<AttentionModel>(*this);
    ar & state_dim;
    ar & p_W;
    ar & p_b;
  }
};
BOOST_CLASS_EXPORT_KEY(EncoderDecoderAttentionModel)


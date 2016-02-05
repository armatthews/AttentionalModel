#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
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
  unsigned input_dim, state_dim, hidden_dim;
  ParameterIndex p_U, p_V, p_W, p_b, p_c;
  Expression U, V, W, b, c;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<StandardAttentionModel, AttentionModel>();
    ar & input_dim;
    ar & state_dim;
    ar & hidden_dim;
    ar & p_U;
    ar & p_V;
    ar & p_W;
    ar & p_b;
    ar & p_c;

  }
};
BOOST_CLASS_EXPORT_KEY(StandardAttentionModel)

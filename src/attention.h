#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "cnn/expr.h"
#include "utils.h"
#include "syntax_tree.h"
#include <stack>
#include <map>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

const bool SYNTAX_PRIOR = true;
const bool DIAGONAL_PRIOR = true;
const bool COVERAGE_PRIOR = true;
const bool SYNTAX_LSTM = false;
const bool SYNTAX_LOG_FEATS = false;

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

  Expression GetAlignmentVector(const vector<Expression>& inputs, const Expression& state, const SyntaxTree* const tree);
  Expression GetContext(const vector<Expression>& inputs, const Expression& state, const SyntaxTree* const tree);

  Expression DiagonalPrior(const vector<Expression>& inputs, unsigned ti);
  Expression SyntaxPrior(const vector<Expression>& inputs, const SyntaxTree* const tree, Expression a);
private:
  Parameter p_U, p_V, p_W, p_b;
  Expression U, V, W, b;
  Expression WI, input_matrix;

  Expression coverage;
  vector<float> source_percentages_v;
  Expression source_percentages;
  Parameter p_length_ratio;
  Expression length_ratio;

  Parameter p_st_w1, p_st_b1, p_st_w2;
  Expression st_w1, st_b1, st_w2;
  LSTMBuilder fwd_expectation_estimator;
  LSTMBuilder rev_expectation_estimator;
  LookupParameter embeddings;
  Parameter p_exp_w, p_exp_b;
  Expression exp_w, exp_b;

  Parameter p_lamb, p_lamb2, p_lamb3;
  Expression lamb, lamb2, lamb3;
  unsigned ti;
  vector<Expression> node_coverage;
  vector<Expression> node_expected_counts;

  void Visit(const SyntaxTree* const parent, const vector<Expression> node_coverage, const vector<Expression> node_expected_counts, vector<vector<Expression>>& node_log_probs);

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<AttentionModel>(*this);
    ar & p_U;
    ar & p_V;
    ar & p_W;
    ar & p_b;
    ar & p_length_ratio;
    ar & p_st_w1;
    ar & p_st_w2;
    ar & p_st_b1;
    ar & fwd_expectation_estimator;
    ar & rev_expectation_estimator;
    ar & embeddings;
    ar & p_exp_w;
    ar & p_exp_b;
    ar & p_lamb;
    ar & p_lamb2;
    ar & p_lamb3;
  }
};
BOOST_CLASS_EXPORT_KEY(StandardAttentionModel)

class SparsemaxAttentionModel : public StandardAttentionModel {
public:
  SparsemaxAttentionModel();
  SparsemaxAttentionModel(Model& model, unsigned input_dim, unsigned state_dim, unsigned hidden_dim);
  Expression GetAlignmentVector(const vector<Expression>& inputs, const Expression& state);
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<StandardAttentionModel>(*this);
  }
};
BOOST_CLASS_EXPORT_KEY(SparsemaxAttentionModel)

/*class ConvolutionalAttentionModel : public AttentionModel {
  ConvolutionalAttentionModel();
  ConvolutionalAttentionModel(Model& model, unsigned input_dim, unsigned state_dim, unsigned hidden_dim, unsigned conv_size);

  void NewGraph(ComputationGraph& cg);
  Expression GetScoreVector(const vector<Expression>& inputs, const Expression& state);
  Expression GetAlignmentVector(const vector<Expression>& inputs, const Expression& state);
  Expression GetContext(const vector<Expression>& inputs, const Expression& state);

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};*/

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
    ar & boost::serialization::base_object<AttentionModel>(*this);
    ar & state_dim;
    ar & p_W;
    ar & p_b;
  }
};
BOOST_CLASS_EXPORT_KEY(EncoderDecoderAttentionModel)


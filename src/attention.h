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

extern bool SYNTAX_PRIOR;
extern bool DIAGONAL_PRIOR;
extern bool COVERAGE_PRIOR;
extern bool USE_FERTILITY;

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
  StandardAttentionModel(Model& model, unsigned vocab_size, unsigned input_dim, unsigned state_dim, unsigned hidden_dim);

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

  // Coverage prior
  Parameter p_coverage_prior_weight;
  Expression coverage_prior_weight;
  Expression coverage;

  // Diagonal Prior
  Parameter p_diagonal_prior_weight;
  Expression diagonal_prior_weight;
  vector<float> source_percentages_v;
  Expression source_percentages;
  Parameter p_length_ratio;
  Expression length_ratio;
  unsigned ti;

  // Syntax Prior
  Parameter p_syntax_prior_weight;
  Expression syntax_prior_weight;
  Parameter p_st_w1, p_st_b1, p_st_w2;
  Expression st_w1, st_b1, st_w2;
  LookupParameter fertilities;
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
    ar & fertilities;
    ar & p_coverage_prior_weight;
    ar & p_diagonal_prior_weight;
    ar & p_syntax_prior_weight;
  }
};
BOOST_CLASS_EXPORT_KEY(StandardAttentionModel)

class SparsemaxAttentionModel : public StandardAttentionModel {
public:
  SparsemaxAttentionModel();
  SparsemaxAttentionModel(Model& model, unsigned vocab_size, unsigned input_dim, unsigned state_dim, unsigned hidden_dim);
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


#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "dynet/dynet.h"
#include "dynet/lstm.h"
#include "dynet/expr.h"
#include "utils.h"
#include "syntax_tree.h"
#include <stack>
#include <map>

class AttentionPrior {
public:
  AttentionPrior(Model& model);
  virtual ~AttentionPrior();
  virtual void NewGraph(ComputationGraph& cg);
  virtual void SetDropout(float rate);
  virtual void NewSentence(const InputSentence* input);

  virtual Expression Compute(const vector<Expression>& inputs, unsigned target_index);
  virtual Expression Compute(const vector<Expression>& inputs, const SyntaxTree* const tree, unsigned target_index);
  virtual void Notify(Expression attention_vector);
protected:
  Parameter p_weight;
  Expression weight;
  ComputationGraph* pcg;

  AttentionPrior();
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & p_weight;
  }
};

class CoveragePrior : public AttentionPrior {
public:
  explicit CoveragePrior(Model& model);
  void NewGraph(ComputationGraph& cg);
  void NewSentence(const InputSentence* input) override;
  Expression Compute(const vector<Expression>& inputs, unsigned target_index) override;
  Expression Compute(const vector<Expression>& inputs, const SyntaxTree* const tree, unsigned target_index) override;
  void Notify(Expression attention_vector) override;
private:
  // TODO: Fertilities
  Expression coverage;

  CoveragePrior();
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<AttentionPrior>(*this);
  }
};
BOOST_CLASS_EXPORT_KEY(CoveragePrior)


class DiagonalPrior : public AttentionPrior {
public:
  explicit DiagonalPrior(Model& model);
  void NewGraph(ComputationGraph& cg) override;
  void NewSentence(const InputSentence* input) override;
  Expression Compute(const vector<Expression>& inputs, unsigned target_index) override;
  Expression Compute(const vector<Expression>& inputs, const SyntaxTree* const tree, unsigned target_index) override;
private:
  vector<float> source_percentages_v;
  Expression source_percentages;
  Parameter p_length_ratio;
  Expression length_ratio;

  DiagonalPrior();
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<AttentionPrior>(*this);
    ar & p_length_ratio;
  }
};
BOOST_CLASS_EXPORT_KEY(DiagonalPrior)

class MarkovPrior : public AttentionPrior {
public:
  explicit MarkovPrior(Model& model, unsigned window_size);
  void NewGraph(ComputationGraph& cg) override;
  void NewSentence(const InputSentence* input) override;
  Expression Compute(const vector<Expression>& inputs, unsigned target_index) override;
  // TODO: Implement for trees
  void Notify(Expression attention_vector) override;

private:
  Parameter p_filter;
  Expression filter;
  Expression prev_attention_vector;
  unsigned window_size;

  MarkovPrior();
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<AttentionPrior>(*this);
    ar & p_filter;
    ar & window_size;
  }
};
BOOST_CLASS_EXPORT_KEY(MarkovPrior)

class SyntaxPrior : public AttentionPrior {
public:
  explicit SyntaxPrior(Model& model);
  void NewGraph(ComputationGraph& cg) override;
  void NewSentence(const InputSentence* input) override;
  Expression Compute(const vector<Expression>& inputs, const SyntaxTree* const tree, unsigned target_index) override;
  void Notify(Expression attention_vector) override;
  void Visit(const SyntaxTree* parent, vector<vector<Expression>>& node_log_probs);
private:
  Parameter p_st_w1, p_st_b1, p_st_w2;
  Expression st_w1, st_b1, st_w2;
  LookupParameter fertilities;
  vector<Expression> node_coverage;
  vector<Expression> node_expected_counts;

  SyntaxPrior();
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<AttentionPrior>(*this);
  }
};
BOOST_CLASS_EXPORT_KEY(SyntaxPrior)

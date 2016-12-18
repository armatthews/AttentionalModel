#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "dynet/dynet.h"
#include "dynet/lstm.h"
#include "dynet/expr.h"
#include "utils.h"
#include "syntax_tree.h"
#include "prior.h"
#include <stack>
#include <map>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

class AttentionModel {
public:
  virtual ~AttentionModel();

  virtual void NewGraph(ComputationGraph& cg) = 0;
  virtual void NewSentence(const InputSentence* input);
  virtual void SetDropout(float rate) {}
  virtual Expression GetScoreVector(const vector<Expression>& inputs, const Expression& state) = 0;
  virtual Expression GetAlignmentVector(const vector<Expression>& inputs, const Expression& state) = 0;
  virtual Expression GetContext(const vector<Expression>& inputs, const Expression& state) = 0;
  virtual void AddPrior(AttentionPrior* prior);

protected:
  vector<AttentionPrior*> priors;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & priors;
  }
};

class StandardAttentionModel : public AttentionModel {
public:
  StandardAttentionModel();
  StandardAttentionModel(Model& model, unsigned input_dim, unsigned state_dim, unsigned hidden_dim, unsigned key_size);

  void NewGraph(ComputationGraph& cg);
  Expression GetScoreVector(const vector<Expression>& inputs, const Expression& state);
  Expression GetAlignmentVector(const vector<Expression>& inputs, const Expression& state);
  Expression GetContext(const vector<Expression>& inputs, const Expression& state);

  Expression GetAlignmentVector(const vector<Expression>& inputs, const Expression& state, const SyntaxTree* const tree);
  Expression GetContext(const vector<Expression>& inputs, const Expression& state, const SyntaxTree* const tree);

private:
  Parameter p_U, p_V, p_W, p_b;
  Expression U, V, W, b;
  Expression WI, input_matrix;
  unsigned target_index;
  unsigned key_size;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<AttentionModel>(*this);
    ar & key_size;
    ar & p_U;
    ar & p_V;
    ar & p_W;
    ar & p_b;
  }
};
BOOST_CLASS_EXPORT_KEY(StandardAttentionModel)

class SparsemaxAttentionModel : public StandardAttentionModel {
public:
  SparsemaxAttentionModel();
  SparsemaxAttentionModel(Model& model, unsigned input_dim, unsigned state_dim, unsigned hidden_dim, unsigned key_size);
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


#pragma once
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "dynet/dynet.h"
#include "dynet/rnn.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"

using namespace dynet;
using namespace std;
using namespace dynet::expr;

struct TreeLSTMBuilder : public RNNBuilder {
public:
  virtual Expression back() const override;
  virtual std::vector<Expression> final_h() const override;
  virtual std::vector<Expression> final_s() const override;
  virtual unsigned num_h0_components() const override;
  virtual void copy(const RNNBuilder & params) override;
  virtual Expression add_input(int id, std::vector<int> children, const Expression& x) = 0;
  std::vector<Expression> get_h(RNNPointer i) const override { assert (false); }
  std::vector<Expression> get_s(RNNPointer i) const override { assert (false); }
  Expression set_s_impl(int prev, const vector<Expression>& s_new) override { assert (false); }
 protected:
  virtual void new_graph_impl(ComputationGraph& cg) override = 0;
  virtual void start_new_sequence_impl(const std::vector<Expression>& h0) override = 0;
  virtual Expression add_input_impl(int prev, const Expression& x) override;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<RNNBuilder>(*this);
  }
};
BOOST_CLASS_EXPORT_KEY(TreeLSTMBuilder)

struct SocherTreeLSTMBuilder : public TreeLSTMBuilder {
  SocherTreeLSTMBuilder() = default;
  explicit SocherTreeLSTMBuilder(unsigned N, //Max branching factor
                       unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model);

  Expression add_input(int id, std::vector<int> children, const Expression& x);
  void copy(const RNNBuilder & params) override;
 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression Lookup(unsigned layer, unsigned p_type, unsigned value);

 public:
  // first index is layer, then ...
  std::vector<std::vector<Parameter>> params;
  std::vector<std::vector<LookupParameter>> lparams;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;
  std::vector<std::vector<std::vector<Expression>>> lparam_vars;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
  unsigned N; // Max branching factor
private:
  ComputationGraph* cg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<TreeLSTMBuilder>(*this);
    ar & params;
    ar & lparams;
    ar & layers;
    ar & N;
  }
};
BOOST_CLASS_EXPORT_KEY(SocherTreeLSTMBuilder)

struct TreeLSTMBuilder2 : public TreeLSTMBuilder {
  TreeLSTMBuilder2() = default;
  explicit TreeLSTMBuilder2(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model);

  Expression add_input(int id, std::vector<int> children, const Expression& x);
 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;

 public:
  LSTMBuilder node_builder;
  std::vector<Expression> h;

private:
  ComputationGraph* cg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<TreeLSTMBuilder>(*this);
    ar & node_builder;
  }
};
BOOST_CLASS_EXPORT_KEY(TreeLSTMBuilder2)

struct BidirectionalTreeLSTMBuilder2 : public TreeLSTMBuilder {
  BidirectionalTreeLSTMBuilder2() = default;
  explicit BidirectionalTreeLSTMBuilder2(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model);

  Expression add_input(int id, std::vector<int> children, const Expression& x);
 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression set_h_impl(int prev, const vector<Expression>& h_new) override;

 public:
  LSTMBuilder fwd_node_builder;
  LSTMBuilder rev_node_builder;
  std::vector<Expression> h;

private:
  ComputationGraph* cg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<TreeLSTMBuilder>(*this);
    ar & fwd_node_builder;
    ar & rev_node_builder;
  }
};
BOOST_CLASS_EXPORT_KEY(BidirectionalTreeLSTMBuilder2)

struct DerpTreeLSTMBuilder : public TreeLSTMBuilder {
  DerpTreeLSTMBuilder() = default;
  explicit DerpTreeLSTMBuilder(Model* model);
  Expression add_input(int id, std::vector<int> children, const Expression& x);
protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
public:
  std::vector<Expression> h;
private:
  ComputationGraph* cg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<TreeLSTMBuilder>(*this);
  }
};
BOOST_CLASS_EXPORT_KEY(DerpTreeLSTMBuilder)


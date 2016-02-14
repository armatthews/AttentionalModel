#ifndef CNN_TREELSTM_H_
#define CNN_TREELSTM_H_

#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"

using namespace cnn::expr;

namespace cnn {

class Model;

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
 protected:
  virtual void new_graph_impl(ComputationGraph& cg) override = 0;
  virtual void start_new_sequence_impl(const std::vector<Expression>& h0) override = 0;
  virtual Expression add_input_impl(int prev, const Expression& x) override;
};

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
};

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
};

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

 public:
  LSTMBuilder fwd_node_builder;
  LSTMBuilder rev_node_builder;
  std::vector<Expression> h;

private:
  ComputationGraph* cg;
};

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
};

} // namespace cnn

#endif

#ifndef CNN_TREELSTM_H_
#define CNN_TREELSTM_H_

#include "cnn/lstm.h"
#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/expr.h"

using namespace cnn::expr;

namespace cnn {

class Model;

struct TreeLSTMBuilder : public RNNBuilder {
  TreeLSTMBuilder() = default;
  explicit TreeLSTMBuilder(unsigned, unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model);

  Expression back() const { return lstm_builder.back(); }
  std::vector<Expression> final_h() const { return lstm_builder.final_h(); }
  std::vector<Expression> final_s() const { return lstm_builder.final_s(); }
  unsigned num_h0_components() const override { return lstm_builder.num_h0_components(); }
  void copy(const RNNBuilder & params) override;
  Expression add_input(std::vector<int> children, const Expression& x);
protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression add_input_impl(int prev, const Expression& x) override;
  Expression LookupParameter(unsigned layer, unsigned p_type, unsigned value);

private:
  std::vector<Expression> h;
  RNNPointer* start_state;
  LSTMBuilder lstm_builder;
  ComputationGraph* cg;
};

} // namespace cnn

#endif

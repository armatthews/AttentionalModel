#include "treelstm.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

TreeLSTMBuilder::TreeLSTMBuilder(unsigned,
                         unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model* model) : lstm_builder(layers, input_dim, hidden_dim, model), cg(nullptr) {}

void TreeLSTMBuilder::new_graph_impl(ComputationGraph& cg) {
  this->cg = &cg;
  start_state = new RNNPointer();
  lstm_builder.new_graph(cg);
}

void TreeLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  lstm_builder.start_new_sequence(hinit);
  *start_state = lstm_builder.state();
}

Expression TreeLSTMBuilder::add_input(vector<int> children, const Expression& x) {
  const RNNPointer& prev = *start_state;
  lstm_builder.add_input(prev, x);
  for (int child : children) {
    assert (child > 0 && (unsigned)child < h.size());
    lstm_builder.add_input(h[child]);
  }
  return lstm_builder.back();
}

Expression TreeLSTMBuilder::add_input_impl(int prev, const Expression& x) {
  assert (false);
  return x;
}

void TreeLSTMBuilder::copy(const RNNBuilder & rnn) {
  assert (false);
}
} // namespace cnn

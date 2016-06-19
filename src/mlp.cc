#include "mlp.h"

MLP::MLP() {}

MLP::MLP(Model& model, unsigned input_size, unsigned hidden_size, unsigned output_size) {
  p_wIH = model.add_parameters({hidden_size, input_size});
  p_wHb = model.add_parameters({hidden_size});
  p_wHO = model.add_parameters({output_size, hidden_size});
  p_wOb = model.add_parameters({output_size});
}

void MLP::NewGraph(ComputationGraph& cg) {
  wIH = parameter(cg, p_wIH);
  wHb = parameter(cg, p_wHb);
  wHO = parameter(cg, p_wHO);
  wOb = parameter(cg, p_wOb);
}

Expression MLP::Feed(Expression input) const {
  Expression h = tanh(affine_transform({wHb, wIH, input}));
  Expression o = affine_transform({wOb, wHO, h});
  return o;
}

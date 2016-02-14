#include "attention.h"
BOOST_CLASS_EXPORT_IMPLEMENT(StandardAttentionModel)
BOOST_CLASS_EXPORT_IMPLEMENT(EncoderDecoderAttentionModel)

AttentionModel::~AttentionModel() {}

StandardAttentionModel::StandardAttentionModel() {}

StandardAttentionModel::StandardAttentionModel(Model& model, unsigned input_dim, unsigned state_dim, unsigned hidden_dim) {
  p_W = model.add_parameters({hidden_dim, input_dim});
  p_V = model.add_parameters({hidden_dim, state_dim});
  p_b = model.add_parameters({hidden_dim, 1});
  p_U = model.add_parameters({1, hidden_dim});
}

void StandardAttentionModel::NewGraph(ComputationGraph& cg) {
  U = parameter(cg, p_U);
  V = parameter(cg, p_V);
  W = parameter(cg, p_W);
  b = parameter(cg, p_b);
  input_matrix.pg = nullptr;
}

Expression StandardAttentionModel::GetScoreVector(const vector<Expression>& inputs, const Expression& state) {
  // The score of an input vector x and state s is:
  // U * tanh(Wx + Vs + b) + c
  // The bias c is unnecessary because we're just going to softmax the result anyway.

  // Normally we would loop over each input x and compute its score individually
  // but we can achieve a ~10% speed up by batching this into matrix-matrix
  // operations.
  // Below, the variable Vsb_n is Vsb broadcasted to size of inputs.
  // Note: Vs + b does not depend on x, so we can pre-compute that quantity once.
  // Similarly W * input_matrix does not change per output-position, so we can compute
  // that quantity and save it until we start working on a new sentence.

  if (input_matrix.pg == nullptr) {
    input_matrix = concatenate_cols(inputs);
    WI = W * input_matrix;
  }
  Expression Vsb = affine_transform({b, V, state});
  Expression Vsb_n = concatenate_cols(vector<Expression>(inputs.size(), Vsb));
  Expression h = tanh(WI + Vsb_n);
  Expression scores = transpose(U * h);
  return scores;
}

Expression StandardAttentionModel::GetAlignmentVector(const vector<Expression>& inputs, const Expression& state) {
  return sparsemax(GetScoreVector(inputs, state));
  //return softmax(GetScoreVector(inputs, state));
}

Expression StandardAttentionModel::GetContext(const vector<Expression>& inputs, const Expression& state) {
  Expression dist = GetAlignmentVector(inputs, state);
  Expression context = input_matrix * dist;
  return context;
}

EncoderDecoderAttentionModel::EncoderDecoderAttentionModel() {}

EncoderDecoderAttentionModel::EncoderDecoderAttentionModel(Model& model, unsigned input_dim, unsigned state_dim) : state_dim(state_dim) {
  p_W = model.add_parameters({state_dim, input_dim});
  p_b = model.add_parameters({state_dim});
}

void EncoderDecoderAttentionModel::NewGraph(ComputationGraph& cg) {
  W = parameter(cg, p_W);
  b = parameter(cg, p_b);
}

Expression EncoderDecoderAttentionModel::GetScoreVector(const vector<Expression>& inputs, const Expression& state) {
  return zeroes(*inputs[0].pg, {(unsigned)inputs.size()});
}

Expression EncoderDecoderAttentionModel::GetAlignmentVector(const vector<Expression>& inputs, const Expression& state) {
  return zeroes(*inputs[0].pg, {(unsigned)inputs.size()});
}

Expression EncoderDecoderAttentionModel::GetContext(const vector<Expression>& inputs, const Expression& state) {
  Expression h0b = pickrange(inputs[0], state_dim / 2, state_dim);
  Expression hNf = pickrange(inputs.back(), 0, state_dim / 2);
  Expression context = concatenate({hNf, h0b});
  return context;
}

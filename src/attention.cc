#include "attention.h"
BOOST_CLASS_EXPORT_IMPLEMENT(StandardAttentionModel)

AttentionModel::~AttentionModel() {}

StandardAttentionModel::StandardAttentionModel() {}

StandardAttentionModel::StandardAttentionModel(Model& model, unsigned input_dim, unsigned state_dim, unsigned hidden_dim) : input_dim(input_dim), state_dim(state_dim), hidden_dim(hidden_dim) {
  InitializeParameters(model);
}

void StandardAttentionModel::InitializeParameters(Model& model) {
  cerr << "Initializing StandardAttentionModel" << endl;
  p_W = model.add_parameters({hidden_dim, input_dim});
  p_V = model.add_parameters({hidden_dim, state_dim});
  p_b = model.add_parameters({hidden_dim, 1});
  p_U = model.add_parameters({1, hidden_dim});
  p_c = model.add_parameters({1, 1});
}

void StandardAttentionModel::NewGraph(ComputationGraph& cg) {
  U = parameter(cg, p_U);
  V = parameter(cg, p_V);
  W = parameter(cg, p_W);
  b = parameter(cg, p_b);
  c = parameter(cg, p_c);
}

Expression StandardAttentionModel::GetScoreVector(const vector<Expression>& inputs, const Expression& state) {
  // The score of an input vector x and state s is:
  // U * tanh(Wx + Vs + b) + c

  // Normally we would loop over each input x and compute its score individually
  // but we can achieve a ~10% speed up by batching this into matrix-matrix
  // operations.
  // Below, the variables Vsb_n and c_n are Vsb and c broadcasted to size of inputs.
  // Note: Vs + b does not depend on x, so we can pre-compute that quantity once.
  Expression input_matrix = concatenate_cols(inputs);
  Expression Vsb = affine_transform({b, V, state});
  Expression Vsb_n = concatenate_cols(vector<Expression>(inputs.size(), Vsb));
  Expression h = tanh(affine_transform({Vsb_n, W, input_matrix}));
  Expression c_n = concatenate_cols(vector<Expression>(inputs.size(), c));
  Expression scores = transpose(affine_transform({c_n, U, h}));
  return scores;
}

Expression StandardAttentionModel::GetAlignmentVector(const vector<Expression>& inputs, const Expression& state) {
  return softmax(GetScoreVector(inputs, state));
}

Expression StandardAttentionModel::GetContext(const vector<Expression>& inputs, const Expression& state) {
  // We then softmax that result to get the alignment distribution
  // and take a weighted sum of the inputs w.r.t. the alignments.

  Expression input_matrix = concatenate_cols(inputs);
  Expression dist = GetAlignmentVector(inputs, state);
  return input_matrix * dist;
}

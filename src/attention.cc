#include "attention.h"
BOOST_CLASS_EXPORT_IMPLEMENT(StandardAttentionModel)
BOOST_CLASS_EXPORT_IMPLEMENT(SparsemaxAttentionModel)
BOOST_CLASS_EXPORT_IMPLEMENT(EncoderDecoderAttentionModel)

bool SYNTAX_PRIOR = false;
bool DIAGONAL_PRIOR = false;
bool COVERAGE_PRIOR = false;
bool USE_FERTILITY = false;

void StandardAttentionModel::Visit(const SyntaxTree* const parent, const vector<Expression> node_coverage, const vector<Expression> node_expected_counts, vector<vector<Expression>>& node_log_probs) {
  for (unsigned i = 0; i < parent->NumChildren(); ++i) {
    const SyntaxTree* const child = &parent->GetChild(i);
    assert (node_log_probs[child->id()].size() == 0);
    node_log_probs[child->id()] = node_log_probs[parent->id()];
  }

  if (parent->NumChildren() < 2) {
    return;
  }

  ComputationGraph& cg = *node_coverage.back().pg;

  Expression parent_coverage = node_coverage[parent->id()];
  Expression parent_expected_count = node_expected_counts[parent->id()];

  vector<Expression> child_scores(parent->NumChildren());
  for (unsigned i = 0; i < parent->NumChildren(); ++i) {
    const SyntaxTree* const child = &parent->GetChild(i);
    Expression child_coverage = node_coverage[child->id()];
    Expression child_expected_count = node_expected_counts[child->id()];
    Expression input = concatenate({parent_coverage, parent_expected_count, child_coverage, child_expected_count});
    input = concatenate({input, log(input + 1e-40)});
    Expression h = tanh(affine_transform({st_b1, st_w1, input}));
    child_scores[i] = st_w2 * h;
  }

  Expression scores = log_softmax(concatenate(child_scores));
  for (unsigned i = 0; i < parent->NumChildren(); ++i) {
    const SyntaxTree* const child = &parent->GetChild(i);
    node_log_probs[child->id()].push_back(pick(scores, i));
  }
}

AttentionModel::~AttentionModel() {}

StandardAttentionModel::StandardAttentionModel() {}

StandardAttentionModel::StandardAttentionModel(Model& model, unsigned vocab_size, unsigned input_dim, unsigned state_dim, unsigned hidden_dim) {
  p_W = model.add_parameters({hidden_dim, input_dim});
  p_V = model.add_parameters({hidden_dim, state_dim});
  p_b = model.add_parameters({hidden_dim, 1});
  p_U = model.add_parameters({1, hidden_dim});
  p_coverage_prior_weight = model.add_parameters({1});
  p_diagonal_prior_weight = model.add_parameters({1});
  p_length_ratio = model.add_parameters({1});
  p_st_w1 = model.add_parameters({hidden_dim, 8});
  p_st_w2 = model.add_parameters({1, hidden_dim});
  p_st_b1 = model.add_parameters({hidden_dim});
  fertilities = model.add_lookup_parameters(vocab_size, {1}); // XXX : Should be vocab size
  p_syntax_prior_weight = model.add_parameters({1});

  p_coverage_prior_weight.get()->values.v[0] = 0.1;
  p_diagonal_prior_weight.get()->values.v[0] = 0.1;
  p_syntax_prior_weight.get()->values.v[0] = 0.1;
}

void StandardAttentionModel::NewGraph(ComputationGraph& cg) {
  U = parameter(cg, p_U);
  V = parameter(cg, p_V);
  W = parameter(cg, p_W);
  b = parameter(cg, p_b);
  coverage_prior_weight = parameter(cg, p_coverage_prior_weight);
  diagonal_prior_weight = parameter(cg, p_diagonal_prior_weight);
  syntax_prior_weight = parameter(cg, p_syntax_prior_weight);
  length_ratio = parameter(cg, p_length_ratio);
  input_matrix.pg = nullptr;
  coverage.pg = nullptr;
  source_percentages.pg = nullptr;
  node_coverage.clear();
  ti = 0;
  st_w1 = parameter(cg, p_st_w1);
  st_w2 = parameter(cg, p_st_w2);
  st_b1 = parameter(cg, p_st_b1);
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

Expression StandardAttentionModel::DiagonalPrior(const vector<Expression>& inputs, unsigned ti) {
  ComputationGraph& cg = *inputs[0].pg;

  if (source_percentages.pg == nullptr) {
    source_percentages_v.resize(inputs.size());
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (i == 0) { // <s>
        source_percentages_v[i] = 0.0;
      }
      else if (i == inputs.size() - 1) { // </s>
       source_percentages_v[i] = 1.0;
      }
      else {
        source_percentages_v[i] = 1.0 * (i - 1) / (inputs.size() - 2);
      }
    }
    source_percentages = input(cg, {(unsigned)inputs.size()}, &source_percentages_v);
  }

  Expression tN = inputs.size() * length_ratio;
  Expression target_percentage = cdiv(input(cg, ti), tN);
  vector<Expression> target_percentage_n(inputs.size(), target_percentage);
  Expression tp = concatenate(target_percentage_n);
  Expression diff = source_percentages - tp;
  // max (x, -x) = abs
  Expression diagonal_prior = softmax(-max(-diff, diff));
  return diagonal_prior;
}

Expression StandardAttentionModel::SyntaxPrior(const vector<Expression>& inputs, const SyntaxTree* const tree, Expression a) {
  if (node_coverage.size() == 0) {
    node_coverage.resize(tree->NumNodes());
    for (unsigned i = 0; i < tree->NumNodes(); ++i) {
      node_coverage[i] = zeroes(*a.pg, {1});
    }

    node_expected_counts.resize(tree->NumNodes());
    for (unsigned i = 0; i < tree->NumNodes(); ++i) {
      if (!USE_FERTILITY) {
        node_expected_counts[i] = input(*a.pg, tree->GetTerminals().size());
      }
      else {
        const Sentence& terminals = tree->GetTerminals();
        vector<Expression> child_fertilities;
        for (WordId w : terminals) {
          Expression fertility = lookup(*a.pg, fertilities, w);
          child_fertilities.push_back(fertility);
        }
        node_expected_counts[i] = sum(child_fertilities);
      }
    }
  }

  vector<const SyntaxTree*> node_stack;
  vector<unsigned> index_stack;
  unsigned terminal_index = 0;
  node_stack.push_back(tree);
  index_stack.push_back(0);

  while (node_stack.size() > 0) {
    const SyntaxTree* node = node_stack.back();
    unsigned index = index_stack.back();
    index_stack.pop_back();
    if (index >= node->NumChildren()) {
      if (node->NumChildren() == 0) {
        Expression c = pick(a, terminal_index);
        for (const SyntaxTree* n : node_stack) {
          node_coverage[n->id()] = node_coverage[n->id()] + c;
        }
        ++terminal_index;
      }
      node_stack.pop_back();
    }
    else {
      index_stack.push_back(index + 1);
      const SyntaxTree* const child = &node->GetChild(index);
      node_stack.push_back(child);
      index_stack.push_back(0);
    }
  }
  assert (node_stack.size() == 0);
  assert (index_stack.size() == 0);

  vector<vector<Expression>> node_log_probs(tree->NumNodes()); // sum(node_log_probs[i]) gives the prior prob for node i. Should sum to 1 over terminals (but not all nodes!)
  vector<const SyntaxTree*> terminals;
  node_stack.push_back(tree);
  index_stack.push_back(0);
  Visit(tree, node_coverage, node_expected_counts, node_log_probs);
  while (node_stack.size() > 0) {
    const SyntaxTree* node = node_stack.back();
    unsigned index = index_stack.back();
    index_stack.pop_back();
    if (index >= node->NumChildren()) {
      if (node->NumChildren() == 0) {
        terminals.push_back(node);
      }
      node_stack.pop_back();
    }
    else {
      index_stack.push_back(index + 1);
      const SyntaxTree* const child = &node->GetChild(index);
      Visit(child, node_coverage, node_expected_counts, node_log_probs);
      node_stack.push_back(child);
      index_stack.push_back(0);
    }
  }

  vector<Expression> terminal_priors(terminals.size());
  for (unsigned i = 0; i < terminals.size(); ++i) {
    const SyntaxTree* terminal = terminals[i];
    assert (node_log_probs[terminal->id()].size() > 0);
    terminal_priors[i] = sum(node_log_probs[terminal->id()]);
  }

  Expression syntax_prior = softmax(concatenate(terminal_priors));
  return syntax_prior;
}

Expression StandardAttentionModel::GetAlignmentVector(const vector<Expression>& inputs, const Expression& state) {
  Expression a = softmax(GetScoreVector(inputs, state));

  if (coverage.pg == nullptr) {
    coverage = zeroes(*a.pg, {(unsigned)inputs.size()});
  }

  if (COVERAGE_PRIOR) {
    Expression coverage_prior = softmax(1 - coverage);
    a = cwise_multiply(a, pow(coverage_prior, coverage_prior_weight));
  }

  if (DIAGONAL_PRIOR) {
    Expression diagonal_prior = DiagonalPrior(inputs, ti);
    a = cwise_multiply(a, pow(diagonal_prior, diagonal_prior_weight));
  }

  if (COVERAGE_PRIOR || DIAGONAL_PRIOR || SYNTAX_PRIOR) {
    Expression Z = sum_cols(transpose(a));
    vector<Expression> Z_n(inputs.size(), Z);
    Expression Zc = concatenate(Z_n);
    a = cdiv(a, Zc);
  }

  coverage = coverage + a;
  ++ti;

  return a;
}

Expression StandardAttentionModel::GetAlignmentVector(const vector<Expression>& inputs, const Expression& state, const SyntaxTree* const tree) {
  Expression a = softmax(GetScoreVector(inputs, state));

  if (coverage.pg == nullptr) {
    coverage = zeroes(*a.pg, {(unsigned)inputs.size()});
  }

  if (COVERAGE_PRIOR) {
    Expression coverage_prior = softmax(1 - coverage);
    a = cwise_multiply(a, pow(coverage_prior, coverage_prior_weight));
  }

  if (DIAGONAL_PRIOR) {
    Expression diagonal_prior = DiagonalPrior(inputs, ti);
    a = cwise_multiply(a, pow(diagonal_prior, diagonal_prior_weight));
  }

  if (SYNTAX_PRIOR) {
    Expression syntax_prior = SyntaxPrior(inputs, tree, a);
    a = cwise_multiply(a, pow(syntax_prior, syntax_prior_weight));
  }

  if (COVERAGE_PRIOR || DIAGONAL_PRIOR || SYNTAX_PRIOR) {
    Expression Z = sum_cols(transpose(a));
    vector<Expression> Z_n(inputs.size(), Z);
    Expression Zc = concatenate(Z_n);
    a = cdiv(a, Zc);
  }

  coverage = coverage + a;
  ++ti;

  return a;
}

Expression StandardAttentionModel::GetContext(const vector<Expression>& inputs, const Expression& state) {
  Expression dist = GetAlignmentVector(inputs, state);
  Expression context = input_matrix * dist;
  return context;
}

Expression StandardAttentionModel::GetContext(const vector<Expression>& inputs, const Expression& state, const SyntaxTree* const tree) {
  Expression dist = GetAlignmentVector(inputs, state, tree);
  Expression context = input_matrix * dist;
  return context;
}

SparsemaxAttentionModel::SparsemaxAttentionModel() : StandardAttentionModel() {}

SparsemaxAttentionModel::SparsemaxAttentionModel(Model& model, unsigned vocab_size, unsigned input_dim, unsigned state_dim, unsigned hidden_dim) : StandardAttentionModel(model, vocab_size, input_dim, state_dim, hidden_dim) {}

Expression SparsemaxAttentionModel::GetAlignmentVector(const vector<Expression>& inputs, const Expression& state) {
  return sparsemax(GetScoreVector(inputs, state));
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

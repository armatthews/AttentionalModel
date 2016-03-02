#include "attention.h"
BOOST_CLASS_EXPORT_IMPLEMENT(StandardAttentionModel)
BOOST_CLASS_EXPORT_IMPLEMENT(SparsemaxAttentionModel)
BOOST_CLASS_EXPORT_IMPLEMENT(EncoderDecoderAttentionModel)

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

StandardAttentionModel::StandardAttentionModel(Model& model, unsigned input_dim, unsigned state_dim, unsigned hidden_dim) {
  p_W = model.add_parameters({hidden_dim, input_dim});
  p_V = model.add_parameters({hidden_dim, state_dim});
  p_b = model.add_parameters({hidden_dim, 1});
  p_U = model.add_parameters({1, hidden_dim});
  p_lamb = model.add_parameters({1});
  p_lamb2 = model.add_parameters({1});
  p_length_ratio = model.add_parameters({1});
  p_st_w1 = model.add_parameters({hidden_dim, 4});
  p_st_w2 = model.add_parameters({1, hidden_dim});
  p_st_b1 = model.add_parameters({hidden_dim});
  fwd_expectation_estimator = LSTMBuilder(2, hidden_dim, hidden_dim / 2, &model);
  rev_expectation_estimator = LSTMBuilder(2, hidden_dim, hidden_dim / 2, &model);
  embeddings = model.add_lookup_parameters(14000, {hidden_dim});
  p_exp_w = model.add_parameters({1, hidden_dim});
  p_exp_b = model.add_parameters({1});
}

void StandardAttentionModel::NewGraph(ComputationGraph& cg) {
  U = parameter(cg, p_U);
  V = parameter(cg, p_V);
  W = parameter(cg, p_W);
  b = parameter(cg, p_b);
  lamb = parameter(cg, p_lamb);
  lamb2 = parameter(cg, p_lamb2);
  length_ratio = parameter(cg, p_length_ratio);
  input_matrix.pg = nullptr;
  coverage.pg = nullptr;
  source_percentages.pg = nullptr;
  node_coverage.clear();
  ti = 0;
  st_w1 = parameter(cg, p_st_w1);
  st_w2 = parameter(cg, p_st_w2);
  st_b1 = parameter(cg, p_st_b1);
  fwd_expectation_estimator.new_graph(cg);
  rev_expectation_estimator.new_graph(cg);
  exp_w = parameter(cg, p_exp_w);
  exp_b = parameter(cg, p_exp_b);
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
  Expression a = softmax(GetScoreVector(inputs, state));

  /*// Coverage stuff
  if (coverage.pg == nullptr) {
    coverage = zeroes(*a.pg, {(unsigned)inputs.size()});
  }
  Expression prior2 = softmax(1 - coverage);

  // Diagonal prior
  // max (x, -x) = abs
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
    source_percentages = input(*a.pg, {(unsigned)inputs.size()}, &source_percentages_v);
  }

  Expression tN = inputs.size() * length_ratio;
  Expression target_percentage = cdiv(input(*a.pg, ti), tN);
  vector<Expression> target_percentage_n(inputs.size(), target_percentage);
  Expression tp = concatenate(target_percentage_n);
  Expression diff = source_percentages - tp;
  Expression prior = softmax(-max(-diff, diff));*/

  // General prior stuff
  /*vector<Expression> vw(inputs.size(), lamb);
  vector<Expression> vv(inputs.size(), 1 - lamb);
  Expression w = concatenate(vw);
  Expression v = concatenate(vv);
  //a = softmax(cwise_multiply(v, log(a)) + cwise_multiply(w, log(prior)));*/
  /*a = cwise_multiply(pow(a, lamb), pow(prior, lamb2) + pow(prior2, 1 - lamb - lamb2));
  Expression Z = sum_cols(transpose(a));
  vector<Expression> Z_n(inputs.size(), Z);
  Expression Zc = concatenate(Z_n);
  a = cdiv(a, Zc);*/

  //coverage = coverage + a;
  //++ti;

  return a;
}

Expression StandardAttentionModel::GetAlignmentVector(const vector<Expression>& inputs, const Expression& state, const SyntaxTree* const tree) {
  Expression a = softmax(GetScoreVector(inputs, state));
  if (node_coverage.size() == 0) {
    node_coverage.resize(tree->NumNodes());
    for (unsigned i = 0; i < tree->NumNodes(); ++i) {
      node_coverage[i] = zeroes(*a.pg, {1});
    }

    map<WordId, Expression> word_embeddings;
    node_expected_counts.resize(tree->NumNodes());
    for (unsigned i = 0; i < tree->NumNodes(); ++i) {
      // Naive: just count terminals
      node_expected_counts[i] = input(*a.pg, tree->GetTerminals().size());

      /*const Sentence& terminals = tree->GetTerminals();
      Expression fwd_exp, rev_exp;

      fwd_expectation_estimator.start_new_sequence();
      for (WordId w : terminals) {
        if (word_embeddings.find(w) == word_embeddings.end()) {
          word_embeddings[w] = lookup(*a.pg, embeddings, w);
        }
        fwd_exp = fwd_expectation_estimator.add_input(word_embeddings[w]);
      }

      rev_expectation_estimator.start_new_sequence();
      for (auto it = terminals.rbegin(); it != terminals.rend(); ++it) {
        WordId w = *it;
        assert (word_embeddings.find(w) != word_embeddings.end());
        rev_exp = rev_expectation_estimator.add_input(word_embeddings[w]);
      }

      Expression expectation_h = concatenate({fwd_exp, rev_exp});
      Expression expectation = affine_transform({exp_b, exp_w, expectation_h});
      node_expected_counts[i] = expectation;*/
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

  Expression prior = softmax(concatenate(terminal_priors));
  vector<Expression> vw(inputs.size(), lamb);
  vector<Expression> vv(inputs.size(), 1 - lamb);
  Expression w = concatenate(vw);
  Expression v = concatenate(vv);
  a = softmax(cwise_multiply(v, log(a)) + cwise_multiply(w, log(prior)));

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

SparsemaxAttentionModel::SparsemaxAttentionModel(Model& model, unsigned input_dim, unsigned state_dim, unsigned hidden_dim) : StandardAttentionModel(model, input_dim, state_dim, hidden_dim) {}

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

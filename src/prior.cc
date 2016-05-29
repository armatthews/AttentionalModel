#include "prior.h"
BOOST_CLASS_EXPORT_IMPLEMENT(CoveragePrior)
BOOST_CLASS_EXPORT_IMPLEMENT(DiagonalPrior)
BOOST_CLASS_EXPORT_IMPLEMENT(MarkovPrior)
BOOST_CLASS_EXPORT_IMPLEMENT(SyntaxPrior)

AttentionPrior::AttentionPrior() {}
AttentionPrior::~AttentionPrior() {}

AttentionPrior::AttentionPrior(Model& model) : pcg(nullptr) {
  p_weight = model.add_parameters({1});
  p_weight.get()->values.v[0] = 0.1;
}

void AttentionPrior::NewGraph(ComputationGraph& cg) {
  weight = parameter(cg, p_weight);
  pcg = &cg;
}

void AttentionPrior::SetDropout(float rate) {}
void AttentionPrior::NewSentence(const Sentence* input) {}

Expression AttentionPrior::Compute(const vector<Expression>& inputs, unsigned target_index) {
  assert (false && "Invalid call to Compute() on a prior that does not accept string inputs");
}

Expression AttentionPrior::Compute(const vector<Expression>& inputs, const SyntaxTree* tree, unsigned target_index) {
  assert (false && "Invalid call to Compute() on a prior that does not accept tree inputs");
}

void AttentionPrior::Notify(Expression attention_vector) {}

CoveragePrior::CoveragePrior() : AttentionPrior() {}

CoveragePrior::CoveragePrior(Model& model) : AttentionPrior(model) {}

void CoveragePrior::NewGraph(ComputationGraph& cg) {
  AttentionPrior::NewGraph(cg);
}

void CoveragePrior::NewSentence(const Sentence* input) {
  const LinearSentence* sent = dynamic_cast<const LinearSentence*>(input);
  coverage = zeroes(*pcg, {(unsigned)sent->size()});
}

Expression CoveragePrior::Compute(const vector<Expression>& inputs, unsigned target_index) {
  Expression prior = softmax(1 - coverage);
  prior.pg->incremental_forward();
  return pow(prior, weight);
}

Expression CoveragePrior::Compute(const vector<Expression>& inputs, const SyntaxTree* tree, unsigned target_index) {
  return Compute(inputs, target_index);
}

void CoveragePrior::Notify(Expression attention_vector) {
  coverage = coverage + attention_vector;
}

DiagonalPrior::DiagonalPrior() : AttentionPrior() {}

DiagonalPrior::DiagonalPrior(Model& model) : AttentionPrior(model) {
  p_length_ratio = model.add_parameters({1});
}

void DiagonalPrior::NewGraph(ComputationGraph& cg) {
  AttentionPrior::NewGraph(cg);
  length_ratio = parameter(cg, p_length_ratio);
}

void DiagonalPrior::NewSentence(const Sentence* translator_input) {
  const LinearSentence* sent = dynamic_cast<const LinearSentence*>(translator_input);
  unsigned source_length = sent->size();
  source_percentages_v.resize(source_length);
  for (unsigned i = 0; i < source_length; ++i) {
    if (i == 0) { // <s>
      source_percentages_v[i] = 0.0;
    }
    else if (i == source_length - 1) { // </s>
     source_percentages_v[i] = 1.0;
    }
    else {
      source_percentages_v[i] = 1.0 * (i - 1) / (source_length - 2);
    }
  }
  source_percentages = input(*pcg, {source_length}, &source_percentages_v);
}

Expression DiagonalPrior::Compute(const vector<Expression>& inputs, unsigned target_index) {
  Expression target_expected_length = inputs.size() * length_ratio;
  Expression target_percentage = cdiv(input(*pcg, target_index), target_expected_length);
  vector<Expression> target_percentage_n(inputs.size(), target_percentage);
  Expression tp = concatenate(target_percentage_n);
  Expression diff = source_percentages - tp;
  // max (x, -x) = abs
  Expression prior = softmax(-max(-diff, diff));
  return pow(prior, weight);
}

Expression DiagonalPrior::Compute(const vector<Expression>& inputs, const SyntaxTree* tree, unsigned target_index) {
  return Compute(inputs, target_index);
}

MarkovPrior::MarkovPrior() : AttentionPrior() {}

MarkovPrior::MarkovPrior(Model& model, unsigned window_size) : AttentionPrior(model), window_size(window_size) {
  p_filter = model.add_parameters({1, window_size});
}

void MarkovPrior::NewGraph(ComputationGraph& cg) {
  AttentionPrior::NewGraph(cg);
  filter = parameter(cg, p_filter);
}

void MarkovPrior::NewSentence(const Sentence* input) {
  const LinearSentence* sent = dynamic_cast<const LinearSentence*>(input);
  prev_attention_vector = zeroes(*pcg, {(unsigned)sent->size()});
}

Expression MarkovPrior::Compute(const vector<Expression>& inputs, unsigned target_index) {
  Expression prior = conv1d_wide(transpose(prev_attention_vector), filter);
  prior = transpose(prior);
  prior = pickrange(prior, 0, inputs.size());
  prior = softmax(prior);
  return pow(prior, weight);
}

void MarkovPrior::Notify(Expression attention_vector) {
  prev_attention_vector = attention_vector;
}

SyntaxPrior::SyntaxPrior() : AttentionPrior() {}

SyntaxPrior::SyntaxPrior(Model& model) : AttentionPrior(model) {
  unsigned hidden_dim = 32; // XXX
  p_st_w1 = model.add_parameters({hidden_dim, 8});
  p_st_w2 = model.add_parameters({1, hidden_dim});
  p_st_b1 = model.add_parameters({hidden_dim});
}

void SyntaxPrior::NewGraph(ComputationGraph& cg) {
  AttentionPrior::NewGraph(cg);
  st_w1 = parameter(cg, p_st_w1);
  st_w2 = parameter(cg, p_st_w2);
  st_b1 = parameter(cg, p_st_b1);
}

void SyntaxPrior::NewSentence(const Sentence* input) {
  // TODO: This needs access to trees
  /*node_coverage.resize(tree->NumNodes());
  for (unsigned i = 0; i < tree->NumNodes(); ++i) {
    node_coverage[i] = zeroes(*a.pg, {1});
  }

  node_expected_counts.resize(tree->NumNodes());
  for (unsigned i = 0; i < tree->NumNodes(); ++i) {
    if (!USE_FERTILITY) {
      node_expected_counts[i] = input(*a.pg, tree->GetTerminals().size());
    }
    else {
      const LinearSentence& terminals = tree->GetTerminals();
      vector<Expression> child_fertilities;
      for (WordId w : terminals) {
        Expression fertility = lookup(*a.pg, fertilities, w);
        child_fertilities.push_back(fertility);
      }
      node_expected_counts[i] = sum(child_fertilities);
    }
  }*/
}

Expression SyntaxPrior::Compute(const vector<Expression>& inputs, const SyntaxTree* tree, unsigned target_index) {
vector<const SyntaxTree*> node_stack;
  vector<unsigned> index_stack;
  unsigned terminal_index = 0;
  node_stack.push_back(tree);
  index_stack.push_back(0);

  // Update node coverage with the current attention vector
  // TODO: This should probably be moved to Notify
  /*while (node_stack.size() > 0) {
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
      const SyntaxTree* child = &node->GetChild(index);
      node_stack.push_back(child);
      index_stack.push_back(0);
    }
  }
  assert (node_stack.size() == 0);
  assert (index_stack.size() == 0);*/

  vector<vector<Expression>> node_log_probs(tree->NumNodes()); // sum(node_log_probs[i]) gives the prior prob for node i. Should sum to 1 over terminals (but not all nodes!)
  vector<const SyntaxTree*> terminals;
  node_stack.push_back(tree);
  index_stack.push_back(0);
  Visit(tree, node_log_probs);
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
      const SyntaxTree* child = &node->GetChild(index);
      Visit(child, node_log_probs);
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
  return pow(syntax_prior, weight);
}

void SyntaxPrior::Notify(Expression attention_vector) {
}

void SyntaxPrior::Visit(const SyntaxTree* parent, vector<vector<Expression>>& node_log_probs) {
  for (unsigned i = 0; i < parent->NumChildren(); ++i) {
    const SyntaxTree* child = &parent->GetChild(i);
    assert (node_log_probs[child->id()].size() == 0);
    node_log_probs[child->id()] = node_log_probs[parent->id()];
  }

  if (parent->NumChildren() < 2) {
    return;
  }

  Expression parent_coverage = node_coverage[parent->id()];
  Expression parent_expected_count = node_expected_counts[parent->id()];

  vector<Expression> child_scores(parent->NumChildren());
  for (unsigned i = 0; i < parent->NumChildren(); ++i) {
    const SyntaxTree* child = &parent->GetChild(i);
    Expression child_coverage = node_coverage[child->id()];
    Expression child_expected_count = node_expected_counts[child->id()];
    Expression input = concatenate({parent_coverage, parent_expected_count, child_coverage, child_expected_count});
    input = concatenate({input, log(input + 1e-40)});
    Expression h = tanh(affine_transform({st_b1, st_w1, input}));
    child_scores[i] = st_w2 * h;
  }

  Expression scores = log_softmax(concatenate(child_scores));
  for (unsigned i = 0; i < parent->NumChildren(); ++i) {
    const SyntaxTree* child = &parent->GetChild(i);
    node_log_probs[child->id()].push_back(pick(scores, i));
  }
}

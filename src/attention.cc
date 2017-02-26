#include "attention.h"
#include "io.h"
BOOST_CLASS_EXPORT_IMPLEMENT(StandardAttentionModel)
BOOST_CLASS_EXPORT_IMPLEMENT(SparsemaxAttentionModel)
BOOST_CLASS_EXPORT_IMPLEMENT(EncoderDecoderAttentionModel)
BOOST_CLASS_EXPORT_IMPLEMENT(TreeAttentionModel)

AttentionModel::~AttentionModel() {}

void AttentionModel::NewSentence(const InputSentence* input) {
  for (AttentionPrior* prior : priors) {
    prior->NewSentence(input);
  }
}

void AttentionModel::AddPrior(AttentionPrior* prior) {
  priors.push_back(prior);
}

StandardAttentionModel::StandardAttentionModel() {}

StandardAttentionModel::StandardAttentionModel(Model& model, unsigned input_dim, unsigned state_dim, unsigned hidden_dim, unsigned key_size) : key_size(key_size) {
  if (key_size == 0) {
    key_size = input_dim;
  }
  assert (key_size <= input_dim);
  p_W = model.add_parameters({hidden_dim, key_size});
  p_V = model.add_parameters({hidden_dim, state_dim});
  p_b = model.add_parameters({hidden_dim, 1});
  p_U = model.add_parameters({1, hidden_dim});
}

void StandardAttentionModel::NewGraph(ComputationGraph& cg) {
  U = parameter(cg, p_U);
  V = parameter(cg, p_V);
  W = parameter(cg, p_W);
  b = parameter(cg, p_b);

  target_index = 0;
  for (AttentionPrior* prior : priors) {
    prior->NewGraph(cg);
  }

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
    vector<Expression> keys(inputs.size());
    for (unsigned i = 0; i < inputs.size(); ++i) {
      keys[i] = pickrange(inputs[i], 0, key_size);
    }
    input_matrix = concatenate_cols(inputs);
    WI = W * concatenate_cols(keys);
  }
  Expression Vsb = affine_transform({b, V, state});
  Expression Vsb_n = concatenate_cols(vector<Expression>(inputs.size(), Vsb));
  Expression h = tanh(WI + Vsb_n);
  Expression scores = transpose(U * h);
  return scores;
}

Expression StandardAttentionModel::GetAlignmentVector(const vector<Expression>& inputs, const Expression& state) {
  Expression a = softmax(GetScoreVector(inputs, state));

  for (AttentionPrior* prior : priors) {
    a = cmult(a, prior->Compute(inputs, target_index));
  }

  // Renormalize if we have priors
  if (priors.size() > 0) {
    Expression Z = sum_cols(transpose(a));
    vector<Expression> Z_n(inputs.size(), Z);
    Expression Zc = concatenate(Z_n);
    a = cdiv(a, Zc);
  }

  ++target_index;

  for (AttentionPrior* prior : priors) {
    prior->Notify(a);
  }

  return a;
}

Expression StandardAttentionModel::GetContext(const vector<Expression>& inputs, const Expression& state) {
  Expression dist = GetAlignmentVector(inputs, state);
  Expression context = input_matrix * dist;
  return context;
}

SparsemaxAttentionModel::SparsemaxAttentionModel() : StandardAttentionModel() {}

SparsemaxAttentionModel::SparsemaxAttentionModel(Model& model, unsigned input_dim, unsigned state_dim, unsigned hidden_dim, unsigned key_size) : StandardAttentionModel(model, input_dim, state_dim, hidden_dim, key_size) {}

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

TreeAttentionModel::TreeAttentionModel() {}

TreeAttentionModel::TreeAttentionModel(Model& model, const SyntaxInputReader* const reader, unsigned input_dim, unsigned tree_dim, unsigned state_dim, unsigned hidden_dim) : state_dim(state_dim), pcg(nullptr) {
  unsigned vocab_size = reader->terminal_vocab.size();
  unsigned label_vocab_size = reader->nonterminal_vocab.size(); 
  tree_encoder = TreeEncoder(model, vocab_size, label_vocab_size, input_dim, input_dim);
  mlp = MLP(model, state_dim + tree_dim, hidden_dim, 1);
}

void TreeAttentionModel::NewGraph(ComputationGraph& cg) {
  tree_encoder.NewGraph(cg);
  mlp.NewGraph(cg);
  target_index = 0;
  pcg = &cg;
}

Expression TreeAttentionModel::GetScoreVector(const vector<Expression>& inputs, const Expression& state) {
  assert (false);
}

Expression TreeAttentionModel::GetAlignmentVector(const vector<Expression>& inputs, const Expression& state) {
  assert (false);
}

Expression TreeAttentionModel::GetContext(const vector<Expression>& inputs, const Expression& state) {
  assert (false);
}

Expression TreeAttentionModel::GetScoreVector(const vector<Expression>& inputs, const SyntaxTree* const tree, const Expression& state) {
  vector<Expression> node_encodings = tree_encoder.Encode(tree);
  vector<Expression> node_scores(node_encodings.size());
  for (unsigned i = 0; i < node_encodings.size(); ++i ) {
    Expression node_encoding = node_encodings[i];
    Expression node_score = mlp.Feed(concatenate({state, node_encoding}));
    node_scores[i] = node_score;
  }
 
  vector<const SyntaxTree*> node_stack = {tree};
  vector<unsigned> ancestor_indices = {};
  vector<unsigned> index_stack = {0};
  unsigned node_index = 0;
  unsigned terminal_index = 0;
  
  vector<Expression> terminal_scores;
  while (node_stack.size() > 0) {
    assert (node_stack.size() == index_stack.size());
    const SyntaxTree* node = node_stack.back();
    const unsigned i = index_stack.back();
    if (i < node->NumChildren()) {
      // The current node still has children,
      // so push the next one onto the stack.
      index_stack[index_stack.size() - 1] += 1;
      node_stack.push_back(&node->GetChild(i));
      index_stack.push_back(0);
      if (i == 0) {
        ancestor_indices.push_back(node_index);
        node_index++; 
      }
    }
    else { 
      if (node->NumChildren() == 0) {
        ancestor_indices.push_back(node_index);
        //cerr << "Term #" << terminal_index << "(node #" << node_index << ")" << " is " << node_stack.size() << " deep:";
        vector<Expression> ancestor_scores;
        ancestor_scores.reserve(ancestor_indices.size());
        for (unsigned x : ancestor_indices) {
          assert (x < node_scores.size());
          //cerr << " " << x;
          ancestor_scores.push_back(node_scores[x]);
        }
        terminal_scores.push_back(sum(ancestor_scores));
        //cerr << endl;
        terminal_index++;
        node_index++;
      }
      index_stack.pop_back();
      node_stack.pop_back();
      assert (ancestor_indices.size() > 0);
      ancestor_indices.pop_back();
    }
  }

  //cerr << "Final node index: " << node_index << endl;
  assert (node_stack.size() == index_stack.size());
  assert (node_stack.size() == 0);
  assert (ancestor_indices.size() == 0);
  Expression scores = concatenate(terminal_scores);
  scores = concatenate({input(*pcg, 0.0f), scores, input(*pcg, 0.0f)}); // add scores for <s> and </s>
  return scores;
}

Expression TreeAttentionModel::GetAlignmentVector(const vector<Expression>& inputs, const SyntaxTree* const tree, const Expression& state) {
  Expression a = softmax(GetScoreVector(inputs, tree, state));

  for (AttentionPrior* prior : priors) {
    a = cmult(a, prior->Compute(inputs, target_index));
  }

  // Renormalize if we have priors
  if (priors.size() > 0) {
    Expression Z = sum_cols(transpose(a));
    vector<Expression> Z_n(inputs.size(), Z);
    Expression Zc = concatenate(Z_n);
    a = cdiv(a, Zc);
  }

  ++target_index;

  for (AttentionPrior* prior : priors) {
    prior->Notify(a);
  }

  return a;
}

Expression TreeAttentionModel::GetContext(const vector<Expression>& inputs, const SyntaxTree* const tree, const Expression& state) {
  Expression dist = GetAlignmentVector(inputs, tree, state);
  Expression context = concatenate_cols(inputs) * dist;
  return context;
}


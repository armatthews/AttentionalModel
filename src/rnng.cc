#include "cnn/expr.h"
#include "rnng.h"

using namespace cnn::expr;
const unsigned lstm_layer_count = 2;

unsigned Action::GetIndex() const {
  if (type == Action::kShift) {
    return 0;
  }
  else if (type == Action::kReduce) {
    return 1;
  }
  else if (type == Action::kNT) {
    return 2 + subtype;
  }
  else {
    assert (false && "Invalid action type");
  }
}


ParserState::ParserState(ParserBuilder* builder) : nopen_parens(0), nt_count(0), prev_action({Action::kNone, 0}), action_count(0), builder(builder) {}

bool ParserState::IsActionForbidden(const Action& a) {
  bool is_shift = (a.type == Action::kShift);
  bool is_reduce = (a.type == Action::kReduce);
  bool is_nt = (a.type == Action::kNT);
  assert(is_shift || is_reduce || is_nt);

  // Don't allow the NT action if the max tree depth has already been reached
  if (is_nt && nopen_parens > kMaxOpenNTs) {
    return true;
  }

  // Allow only the NT action if the only thing on the stack is the guard
  if (stack.size() == 1) {
    return !is_nt;
  }

  // you can't reduce after an NT action
  if (is_reduce && prev_action.type == Action::kNT) {
    return true;
  }
  return false;
}

vector<unsigned> ParserState::GetValidActionList() {
  vector<unsigned> valid_actions;
  if (!IsActionForbidden({Action::kShift, 0})) {
    valid_actions.push_back(0);
  }
  if (!IsActionForbidden({Action::kReduce, 0})) {
    valid_actions.push_back(1);
  }
  if (!IsActionForbidden({Action::kNT, 0})) {
    for (unsigned i = 0; i < builder->nt_vocab_size; ++i) {
      valid_actions.push_back(2 + i);
    }
  }
  return valid_actions;
}

void ParserState::PerformShift(WordId wordid) {
  Expression word = lookup(*builder->pcg, builder->p_w, wordid);
  terms.push_back(word);
  builder->term_lstm.add_input(word);
  stack.push_back(word);
  builder->stack_lstm.add_input(word);
  is_open_paren.push_back(-1);
}

void ParserState::PerformNT(WordId ntid) {
  ++nopen_parens;
  ++nt_count;
  Expression nt_embedding = lookup(*builder->pcg, builder->p_nt, ntid);
  stack.push_back(nt_embedding);
  builder->stack_lstm.add_input(nt_embedding);
  is_open_paren.push_back(ntid);
}

void ParserState::PerformReduce() {
  --nopen_parens;
  // We should have the stack guard plus the two nodess we're about to combine
  assert (stack.size() > 2);
  int last_nt_index = is_open_paren.size() - 1;
  while (is_open_paren[last_nt_index] < 0) {
    --last_nt_index;
    assert (last_nt_index >= 0);
  }

  int nchildren = is_open_paren.size() - last_nt_index - 1;
  assert (nchildren > 0);

  vector<Expression> children(nchildren);
  is_open_paren.pop_back(); // nt symbol
  stack.pop_back(); // nonterminal dummy
  builder->stack_lstm.rewind_one_step(); // nt symbol

  for (unsigned i = 0; i < nchildren; ++i) {
    children[i] = stack.back();
    stack.pop_back();
    builder->stack_lstm.rewind_one_step();
    is_open_paren.pop_back();
  }

  Expression composed = builder->EmbedNonterminal(is_open_paren[last_nt_index], children);
  builder->stack_lstm.add_input(composed);
  stack.push_back(composed);
  is_open_paren.push_back(-1); // we just closed a paren at this position
}

ParserBuilder::ParserBuilder() {}

// TODO: Isn't action_vocab_size = nt_vocab_size + 2 (i.e. SHIFT, REDUCE, and one NT action per NT type)
ParserBuilder::ParserBuilder(Model& model, SoftmaxBuilder* cfsm,
    unsigned vocab_size, unsigned nt_vocab_size, unsigned action_vocab_size, unsigned hidden_dim,
    unsigned term_emb_dim, unsigned nt_emb_dim, unsigned action_emb_dim) :
  stack_lstm(lstm_layer_count, nt_emb_dim, hidden_dim, &model),
  term_lstm(lstm_layer_count, term_emb_dim, hidden_dim, &model),
  action_lstm(lstm_layer_count, action_emb_dim, hidden_dim, &model),
  const_lstm_fwd(1, nt_emb_dim, nt_emb_dim, &model),
  const_lstm_rev(1, nt_emb_dim, nt_emb_dim, &model),

  p_w(model.add_lookup_parameters(vocab_size, {term_emb_dim})),
  p_nt(model.add_lookup_parameters(nt_vocab_size, {nt_emb_dim})),
  p_ntup(model.add_lookup_parameters(nt_vocab_size, {nt_emb_dim})),
  p_a(model.add_lookup_parameters(action_vocab_size, {action_emb_dim})),

  p_pbias(model.add_parameters({hidden_dim})),
  p_A(model.add_parameters({hidden_dim, hidden_dim})),
  p_S(model.add_parameters({hidden_dim, hidden_dim})),
  p_T(model.add_parameters({hidden_dim, hidden_dim})),
  p_cW(model.add_parameters({nt_emb_dim, nt_emb_dim * 2})),
  p_cbias(model.add_parameters({nt_emb_dim})),
  p_p2a(model.add_parameters({action_vocab_size, hidden_dim})),
  p_action_start(model.add_parameters({action_emb_dim})),
  p_abias(model.add_parameters({action_vocab_size})),
  p_stack_guard(model.add_parameters({nt_emb_dim})),

  cfsm(cfsm), dropout_rate(0.0f), nt_vocab_size(nt_vocab_size) {}

void ParserBuilder::SetDropout(float rate) {
  dropout_rate = rate;
  stack_lstm.set_dropout(rate);
  term_lstm.set_dropout(rate);
  action_lstm.set_dropout(rate);
  const_lstm_fwd.set_dropout(rate);
  const_lstm_rev.set_dropout(rate);
}

void ParserBuilder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;

  term_lstm.new_graph(cg);
  stack_lstm.new_graph(cg);
  action_lstm.new_graph(cg);
  const_lstm_fwd.new_graph(cg);
  const_lstm_rev.new_graph(cg);

  cfsm->new_graph(cg);

  pbias = parameter(cg, p_pbias);
  S = parameter(cg, p_S);
  A = parameter(cg, p_A);
  T = parameter(cg, p_T);
  cW = parameter(cg, p_cW);
  cbias = parameter(cg, p_cbias);
  p2a = parameter(cg, p_p2a);
  abias = parameter(cg, p_abias);
  action_start = parameter(cg, p_action_start);
  stack_guard = parameter(cg, p_stack_guard);
}

vector<Action> ParserBuilder::Sample(const vector<WordId>& sentence) {
  assert (false);
}

vector<Action> ParserBuilder::Predict(const vector<WordId>& sentence) {
  assert (false);
};

Expression ParserBuilder::Summarize(const LSTMBuilder& builder) const {
  Expression summary = builder.back();
  if (dropout_rate != 0.0f) {
    summary = dropout(summary, dropout_rate);
  }
  return summary;
}

Expression ParserBuilder::BuildGraph(const vector<Action>& correct_actions) {
  ParserState state(this);
  vector<Expression> neg_log_probs;

  term_lstm.start_new_sequence();
  stack_lstm.start_new_sequence();
  action_lstm.start_new_sequence();

  WordId kSOS = 1; // XXX
  Expression SOS_embedding = lookup(*pcg, p_w, kSOS);
  state.terms.push_back(SOS_embedding);
  term_lstm.add_input(state.terms.back());

  state.stack.push_back(stack_guard);
  stack_lstm.add_input(state.stack.back());
  state.is_open_paren.push_back(-1);

  action_lstm.add_input(action_start);

  unsigned action_count = 0;
  while (state.stack.size() > 2 || state.terms.size() - 1 == 0) {
    vector<unsigned> current_valid_actions = state.GetValidActionList();
    Expression stack_summary = Summarize(stack_lstm);
    Expression action_summary = Summarize(action_lstm);
    Expression term_summary = Summarize(term_lstm);

    Expression p_t = affine_transform({pbias, S, stack_summary, A, action_summary, T, term_summary});
    Expression nlp_t = rectify(p_t);
    Expression r_t = affine_transform({abias, p2a, nlp_t});
    Expression adist = log_softmax(r_t, current_valid_actions);

    assert (action_count < correct_actions.size());
    Action action = correct_actions[action_count];
    Expression neg_log_prob = -pick(adist, action.type);
    neg_log_probs.push_back(neg_log_prob);

    // Perform action
    if (action.type == Action::kShift) {
      WordId wordid = action.subtype;
      Expression loss = cfsm->neg_log_softmax(nlp_t, wordid);
      neg_log_probs.push_back(loss);
      state.PerformShift(wordid);
    }
    else if (action.type == Action::kNT) {
      WordId ntid = action.subtype;
      state.PerformNT(ntid);
    }
    else if (action.type == Action::kReduce) {
      state.PerformReduce();
    }
    else {
      assert (false && "Invalid action!");
    }

    Expression actione = lookup(*pcg, p_a, action.GetIndex());
    action_lstm.add_input(actione);
    action_count++;
  }

  assert (action_count == correct_actions.size());
  assert (state.stack.size() == 2); // guard symbol, root
  return sum(neg_log_probs);
}

Expression ParserBuilder::EmbedNonterminal(WordId nt, const vector<Expression>& children) {
  Expression nt_embedding = lookup(*pcg, p_ntup, nt);
  unsigned nchildren = children.size();
  const_lstm_fwd.start_new_sequence();
  const_lstm_rev.start_new_sequence();
  const_lstm_fwd.add_input(nt_embedding);
  const_lstm_rev.add_input(nt_embedding);

  for (unsigned i = 0; i < nchildren; ++i) {
    const_lstm_fwd.add_input(children[i]);
    const_lstm_rev.add_input(children[nchildren - i - 1]);
  }
  Expression cfwd = Summarize(const_lstm_fwd);
  Expression crev = Summarize(const_lstm_rev);
  Expression c = concatenate({cfwd, crev});
  Expression composed = rectify(affine_transform({cbias, cW, c}));
  return composed;
}


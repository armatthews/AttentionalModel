#include "dynet/expr.h"
#include "rnng.h"
BOOST_CLASS_EXPORT_IMPLEMENT(SourceConditionedParserBuilder)

using namespace dynet::expr;
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

bool Action::operator==(const Action& o) const {
  return type == o.type && subtype == o.subtype;
}


ParserState::ParserState() : nopen_parens(0), prev_action({Action::kNone, 0}) {}

bool ParserState::IsActionForbidden(const Action& a) const {
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

vector<unsigned> ParserBuilder::GetValidActionList() const {
  vector<unsigned> valid_actions;
  if (!curr_state->IsActionForbidden({Action::kShift, 0})) {
    valid_actions.push_back(0);
  }
  if (!curr_state->IsActionForbidden({Action::kReduce, 0})) {
    valid_actions.push_back(1);
  }
  if (!curr_state->IsActionForbidden({Action::kNT, 0})) {
    for (unsigned i = 0; i < nt_vocab_size; ++i) {
      valid_actions.push_back(2 + i);
    }
  }
  return valid_actions;
}

void ParserBuilder::PerformShift(WordId wordid) {
  Expression word = lookup(*pcg, p_w, wordid);
  term_lstm.add_input(word);
  stack_lstm.add_input(word);

  curr_state->terms.push_back(word);
  curr_state->stack.push_back(word);
  curr_state->is_open_paren.push_back(-1);
}

void ParserBuilder::PerformNT(WordId ntid) {
  ++curr_state->nopen_parens;
  Expression nt_embedding = lookup(*pcg, p_nt, ntid);
  stack_lstm.add_input(nt_embedding);

  curr_state->stack.push_back(nt_embedding);
  curr_state->is_open_paren.push_back(ntid);
}

void ParserBuilder::PerformReduce() {
  --curr_state->nopen_parens;
  // We should have the stack guard plus the two nodess we're about to combine
  assert (curr_state->stack.size() > 2);
  int last_nt_index = curr_state->is_open_paren.size() - 1;
  while (curr_state->is_open_paren[last_nt_index] < 0) {
    --last_nt_index;
    assert (last_nt_index >= 0);
  }

  assert (last_nt_index >= 0);
  assert ((unsigned)last_nt_index + 1 <= curr_state->is_open_paren.size());
  unsigned nchildren = curr_state->is_open_paren.size() - last_nt_index - 1;

  vector<Expression> children(nchildren);
  curr_state->is_open_paren.pop_back(); // nt symbol
  curr_state->stack.pop_back(); // nonterminal dummy
  stack_lstm.rewind_one_step(); // nt symbol

  for (unsigned i = 0; i < nchildren; ++i) {
    children[i] = curr_state->stack.back();
    curr_state->stack.pop_back();
    curr_state->is_open_paren.pop_back();
    stack_lstm.rewind_one_step();
  }

  Expression composed = EmbedNonterminal(curr_state->is_open_paren[last_nt_index], children);
  stack_lstm.add_input(composed);
  curr_state->stack.push_back(composed);
  curr_state->is_open_paren.push_back(-1); // we just closed a paren at this position
}

ParserBuilder::ParserBuilder() : curr_state(nullptr) {}

// TODO: Isn't action_vocab_size = nt_vocab_size + 2 (i.e. SHIFT, REDUCE, and one NT action per NT type)
ParserBuilder::ParserBuilder(Model& model, SoftmaxBuilder* cfsm,
    unsigned vocab_size, unsigned nt_vocab_size, unsigned action_vocab_size, unsigned hidden_dim,
    unsigned term_emb_dim, unsigned nt_emb_dim, unsigned action_emb_dim) :
  pcg(nullptr), curr_state(nullptr),

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

Expression ParserBuilder::Summarize(const LSTMBuilder& builder) const {
  Expression summary = builder.back();
  if (dropout_rate != 0.0f) {
    summary = dropout(summary, dropout_rate);
  }
  return summary;
}

Expression ParserBuilder::Summarize(const LSTMBuilder& builder, RNNPointer p) const {
  Expression summary = builder.get_h(p).back();
  if (dropout_rate != 0.0f) {
    summary = dropout(summary, dropout_rate);
  }
  return summary;
}

void ParserBuilder::NewSentence() {
  const WordId kSOS = 1; // XXX
  Expression SOS_embedding = lookup(*pcg, p_w, kSOS);

  prev_states.clear();
  prev_states.push_back(ParserState());
  curr_state = &prev_states.back();

  curr_state->stack.clear();
  curr_state->terms.clear();
  curr_state->is_open_paren.clear();
  curr_state->nopen_parens = 0;
  curr_state->prev_action = {Action::kNone, 0};

  curr_state->terms.push_back(SOS_embedding);
  curr_state->stack.push_back(stack_guard);
  curr_state->is_open_paren.push_back(-1);

  term_lstm.start_new_sequence();
  stack_lstm.start_new_sequence();
  action_lstm.start_new_sequence();

  assert (term_lstm.state() == -1);
  assert (stack_lstm.state() == -1);

  stack_lstm.add_input(curr_state->stack.back());
  term_lstm.add_input(curr_state->terms.back());
  action_lstm.add_input(action_start);

  assert (term_lstm.state() == 0);
  assert (stack_lstm.state() == 0);

  curr_state->stack_lstm_pointer = stack_lstm.state();
  curr_state->terminal_lstm_pointer = term_lstm.state();
  curr_state->action_lstm_pointer = action_lstm.state();

  assert (this->state() == stack_lstm.state());
  assert (this->state() == term_lstm.state());
  assert (this->state() == action_lstm.state());
}

Expression ParserBuilder::GetStateVector() const {
  return GetStateVector(state());
}

Expression ParserBuilder::GetStateVector(RNNPointer p) const {
  assert (p >= 0 && (unsigned)p < prev_states.size());
  ParserState state = prev_states[p];
  Expression stack_summary = Summarize(stack_lstm, state.stack_lstm_pointer);
  Expression action_summary = Summarize(action_lstm, state.action_lstm_pointer);
  Expression term_summary = Summarize(term_lstm, state.terminal_lstm_pointer);

  Expression p_t = affine_transform({pbias, S, stack_summary, A, action_summary, T, term_summary});
  Expression nlp_t = rectify(p_t);
  return nlp_t;
}

Expression ParserBuilder::GetActionDistribution(Expression state_vector) const {
  Expression r_t = affine_transform({abias, p2a, state_vector});
  Expression adist = log_softmax(r_t, GetValidActionList());
  return adist;
}

RNNPointer ParserBuilder::state() const {
  return (RNNPointer)((int)prev_states.size() - 1);
}

void ParserBuilder::PerformAction(const Action& action) {
  assert (prev_states.size() > 0);
  assert (curr_state == &prev_states.back());
  return PerformAction(action, *curr_state);
}

void ParserBuilder::PerformAction(const Action& action, RNNPointer p) {
  assert ((unsigned)p < prev_states.size());
  return PerformAction(action, prev_states[p]);
}

void ParserBuilder::PerformAction(const Action& action, const ParserState& state) {
  prev_states.push_back(state);
  curr_state = &prev_states.back();

  if (action.type == Action::kShift) {
    WordId wordid = action.subtype;
    PerformShift(wordid);
  }
  else if (action.type == Action::kNT) {
    WordId ntid = action.subtype;
    PerformNT(ntid);
  }
  else if (action.type == Action::kReduce) {
    PerformReduce();
  }
  else {
    assert (false && "Invalid action!");
  }

  Expression actione = lookup(*pcg, p_a, action.GetIndex());
  action_lstm.add_input(actione);

  curr_state->prev_action = action;
  curr_state->stack_lstm_pointer = stack_lstm.state();
  curr_state->terminal_lstm_pointer = term_lstm.state();
  curr_state->action_lstm_pointer = action_lstm.state();
}

Expression ParserBuilder::Loss(Expression state_vector, const Action& ref) const {
  Expression action_dist = GetActionDistribution(state_vector);
  Expression neg_log_prob = -pick(action_dist, ref.GetIndex());
  if (ref.type == Action::kShift) {
    WordId wordid = ref.subtype;
    Expression loss = cfsm->neg_log_softmax(state_vector, wordid);
    neg_log_prob = neg_log_prob + loss;
  }
  return neg_log_prob;
}

Expression ParserBuilder::BuildGraph(const vector<Action>& correct_actions) {
  NewSentence();

  vector<Expression> neg_log_probs;
  for (Action action : correct_actions) {
    assert (curr_state->stack.size() > 2 || curr_state->terms.size() - 1 == 0);

    Expression state_vector = GetStateVector();
    Expression neg_log_prob = Loss(state_vector, action);
    neg_log_probs.push_back(neg_log_prob);
    PerformAction(action);
  }

  assert (curr_state->stack.size() == 2); // guard symbol, root
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

SourceConditionedParserBuilder::SourceConditionedParserBuilder() {}

SourceConditionedParserBuilder::SourceConditionedParserBuilder(Model& model, SoftmaxBuilder* cfsm,
    unsigned vocab_size, unsigned nt_vocab_size, unsigned action_vocab_size, unsigned hidden_dim,
    unsigned term_emb_dim, unsigned nt_emb_dim, unsigned action_emb_dim, unsigned source_dim) :
  ParserBuilder(model, cfsm, vocab_size, nt_vocab_size, action_vocab_size, hidden_dim, term_emb_dim,
    nt_emb_dim, action_emb_dim),
  p_W(model.add_parameters({hidden_dim, source_dim})) {}

void SourceConditionedParserBuilder::NewGraph(ComputationGraph& cg) {
  ParserBuilder::NewGraph(cg);
  W = parameter(cg, p_W);
}

Expression SourceConditionedParserBuilder::GetStateVector(Expression source_context) const {
  return GetStateVector(source_context, state());
  Expression stack_summary = Summarize(stack_lstm);
  Expression action_summary = Summarize(action_lstm);
  Expression term_summary = Summarize(term_lstm);

  Expression p_t = affine_transform({pbias, S, stack_summary, A, action_summary, T, term_summary, W, source_context});
  Expression nlp_t = rectify(p_t);
  return nlp_t;
}

Expression SourceConditionedParserBuilder::GetStateVector(Expression source_context, RNNPointer p) const {
  assert (p >= 0 && (unsigned)p < prev_states.size());
  ParserState state = prev_states[p];
  Expression stack_summary = Summarize(stack_lstm, state.stack_lstm_pointer);
  Expression action_summary = Summarize(action_lstm, state.action_lstm_pointer);
  Expression term_summary = Summarize(term_lstm, state.terminal_lstm_pointer);

  Expression p_t = affine_transform({pbias, S, stack_summary, A, action_summary, T, term_summary, W, source_context});
  Expression nlp_t = rectify(p_t);
  return nlp_t;
}


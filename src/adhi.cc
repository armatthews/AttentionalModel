#include "dynet/expr.h"
#include "adhi.h"
#include "utils.h"
#include "io.h"
BOOST_CLASS_EXPORT_IMPLEMENT(SourceConditionedAdhiParserBuilder)

using namespace dynet::expr;
const unsigned lstm_layer_count = 2;

AdhiParserState::AdhiParserState() : nopen_parens(0), prev_action({Action::kNone, 0}) {}

bool AdhiParserState::IsActionForbidden(Action::ActionType at) const {
  bool is_shift = (at == Action::kShift);
  bool is_reduce = (at == Action::kReduce);
  bool is_nt = (at == Action::kNT);
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

vector<unsigned> AdhiParserBuilder::GetValidActionList() const {
  return GetValidActionList(state());
}

vector<unsigned> AdhiParserBuilder::GetValidActionList(RNNPointer p) const {
  vector<unsigned> valid_actions;
  const AdhiParserState& state = prev_states[p];
  if (!state.IsActionForbidden(Action::kShift)) {
    valid_actions.push_back(0);
  }
  if (!state.IsActionForbidden(Action::kReduce)) {
    valid_actions.push_back(1);
  }
  if (!state.IsActionForbidden(Action::kNT)) {
    for (unsigned i = 0; i < nt_vocab_size; ++i) {
      valid_actions.push_back(2 + i);
    }
  }
  return valid_actions;
}

void AdhiParserBuilder::PerformShift(WordId wordid) {
  Expression word = lookup(*pcg, p_w, wordid);
  stack_lstm.add_input(curr_state->stack_lstm_pointer, word);

  curr_state->stack_lstm_pointer = stack_lstm.state();

  curr_state->terms.push_back(word);
  curr_state->stack.push_back(word);
  curr_state->is_open_paren.push_back(-1);
}

void AdhiParserBuilder::PerformNT(WordId ntid) {
  ++curr_state->nopen_parens;
  Expression nt_embedding = lookup(*pcg, p_nt, ntid);
  stack_lstm.add_input(curr_state->stack_lstm_pointer, nt_embedding);

  curr_state->stack_lstm_pointer = stack_lstm.state();
  curr_state->stack.push_back(nt_embedding);
  curr_state->is_open_paren.push_back(ntid);
}

void AdhiParserBuilder::PerformReduce() {
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
  curr_state->stack_lstm_pointer = stack_lstm.get_head(curr_state->stack_lstm_pointer);

  for (unsigned i = 0; i < nchildren; ++i) {
    children[i] = curr_state->stack.back();
    curr_state->stack.pop_back();
    curr_state->is_open_paren.pop_back();
    curr_state->stack_lstm_pointer = stack_lstm.get_head(curr_state->stack_lstm_pointer);
  }

  Expression composed = EmbedNonterminal(curr_state->is_open_paren[last_nt_index], children);
  stack_lstm.add_input(curr_state->stack_lstm_pointer, composed);
  curr_state->stack_lstm_pointer = stack_lstm.state();
  curr_state->stack.push_back(composed);
  curr_state->is_open_paren.push_back(-1); // we just closed a paren at this position
}

AdhiParserBuilder::AdhiParserBuilder() : curr_state(nullptr) {}

// TODO: Isn't action_vocab_size = nt_vocab_size + 2 (i.e. SHIFT, REDUCE, and one NT action per NT type)
AdhiParserBuilder::AdhiParserBuilder(Model& model, SoftmaxBuilder* cfsm,
    unsigned vocab_size, unsigned nt_vocab_size, unsigned action_vocab_size, unsigned hidden_dim,
    unsigned term_emb_dim, unsigned nt_emb_dim) :
  pcg(nullptr), curr_state(nullptr),

  stack_lstm(lstm_layer_count, nt_emb_dim, hidden_dim, model),
  const_lstm_fwd(1, nt_emb_dim, nt_emb_dim, model),
  const_lstm_rev(1, nt_emb_dim, nt_emb_dim, model),

  p_w(model.add_lookup_parameters(vocab_size, {term_emb_dim})),
  p_nt(model.add_lookup_parameters(nt_vocab_size, {nt_emb_dim})),
  p_ntup(model.add_lookup_parameters(nt_vocab_size, {nt_emb_dim})),

  p_pbias(model.add_parameters({hidden_dim})),
  p_S(model.add_parameters({hidden_dim, hidden_dim})),
  p_cW(model.add_parameters({nt_emb_dim, nt_emb_dim * 2})),
  p_cbias(model.add_parameters({nt_emb_dim})),
  p_p2a(model.add_parameters({action_vocab_size, hidden_dim})),
  p_abias(model.add_parameters({action_vocab_size})),
  p_stack_guard(model.add_parameters({nt_emb_dim})),

  cfsm(cfsm), dropout_rate(0.0f), nt_vocab_size(nt_vocab_size) {cerr << "Creating an Adhi thing..." << endl;}

void AdhiParserBuilder::SetDropout(float rate) {
  dropout_rate = rate;
  stack_lstm.set_dropout(rate);
  const_lstm_fwd.set_dropout(rate);
  const_lstm_rev.set_dropout(rate);
}

void AdhiParserBuilder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;

  stack_lstm.new_graph(cg);
  const_lstm_fwd.new_graph(cg);
  const_lstm_rev.new_graph(cg);

  cfsm->new_graph(cg);

  pbias = parameter(cg, p_pbias);
  S = parameter(cg, p_S);
  cW = parameter(cg, p_cW);
  cbias = parameter(cg, p_cbias);
  p2a = parameter(cg, p_p2a);
  abias = parameter(cg, p_abias);
  stack_guard = parameter(cg, p_stack_guard);
}

Expression AdhiParserBuilder::Summarize(const LSTMBuilder& builder) const {
  Expression summary = builder.back();
  if (dropout_rate != 0.0f) {
    summary = dropout(summary, dropout_rate);
  }
  return summary;
}

Expression AdhiParserBuilder::Summarize(const LSTMBuilder& builder, RNNPointer p) const {
  Expression summary = builder.get_h(p).back();
  if (dropout_rate != 0.0f) {
    summary = dropout(summary, dropout_rate);
  }
  return summary;
}

void AdhiParserBuilder::NewSentence() {
  const WordId kSOS = 1; // XXX: This may be wildly broken since now we don't even have SOS in our vocab... Maybe train a separate embedding for this...
  Expression SOS_embedding = lookup(*pcg, p_w, kSOS);

  prev_states.clear();
  prev_states.push_back(AdhiParserState());
  curr_state = &prev_states.back();

  curr_state->stack.clear();
  curr_state->terms.clear();
  curr_state->is_open_paren.clear();
  curr_state->nopen_parens = 0;
  curr_state->prev_action = {Action::kNone, 0};

  curr_state->terms.push_back(SOS_embedding);
  curr_state->stack.push_back(stack_guard);
  curr_state->is_open_paren.push_back(-1);

  stack_lstm.start_new_sequence();

  assert (stack_lstm.state() == -1);

  stack_lstm.add_input(curr_state->stack.back());

  assert (stack_lstm.state() == 0);

  curr_state->stack_lstm_pointer = stack_lstm.state();

  assert (this->state() == stack_lstm.state());
}

Expression AdhiParserBuilder::GetStateVector() const {
  return GetStateVector(state());
}

Expression AdhiParserBuilder::GetStateVector(RNNPointer p) const {
  assert (p >= 0 && (unsigned)p < prev_states.size());
  AdhiParserState state = prev_states[p];
  Expression stack_summary = Summarize(stack_lstm, state.stack_lstm_pointer);

  Expression p_t = affine_transform({pbias, S, stack_summary});
  Expression nlp_t = rectify(p_t);
  return nlp_t;
}

Expression AdhiParserBuilder::GetActionDistribution(Expression state_vector) const {
  return GetActionDistribution(state(), state_vector);
}

Expression AdhiParserBuilder::GetActionDistribution(RNNPointer p, Expression state_vector) const {
  Expression r_t = affine_transform({abias, p2a, state_vector});
  Expression adist = log_softmax(r_t, GetValidActionList(p));
  return adist;
}

RNNPointer AdhiParserBuilder::state() const {
  return (RNNPointer)((int)prev_states.size() - 1);
}

void AdhiParserBuilder::PerformAction(const Action& action) {
  assert (prev_states.size() > 0);
  assert (curr_state == &prev_states.back());
  return PerformAction(action, *curr_state);
}

void AdhiParserBuilder::PerformAction(const Action& action, RNNPointer p) {
  assert ((unsigned)p < prev_states.size());
  return PerformAction(action, prev_states[p]);
}

void AdhiParserBuilder::PerformAction(const Action& action, const AdhiParserState& state) {
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

  curr_state->prev_action = action;
}

Action AdhiParserBuilder::Sample(Expression state_vector) const {
  assert(false);
  return Sample(state(), state_vector);
}

Action AdhiParserBuilder::Sample(RNNPointer p, Expression state_vector) const {
  Expression action_dist = GetActionDistribution(p, state_vector);

  vector<float> dist = as_vector(softmax(action_dist).value());
  unsigned s = ::Sample(dist);

  Action r = convert(s);
  if (r.type == Action::kShift) {
    r.subtype = cfsm->sample(state_vector);
  }
  return r;
}

KBestList<Action> AdhiParserBuilder::PredictKBest(Expression state_vector, unsigned K) const {
  return PredictKBest(state(), state_vector, K);
}

KBestList<Action> AdhiParserBuilder::PredictKBest(RNNPointer p, Expression state_vector, unsigned K) const {
  Expression action_dist = GetActionDistribution(p, state_vector);
  Expression word_dist = cfsm->full_log_distribution(state_vector);

  vector<float> type_dist = as_vector(softmax(action_dist).value());
  vector<float> subtype_dist = as_vector(softmax(word_dist).value());

  KBestList<Action> kbest(K);
  vector<unsigned> valid_actions = GetValidActionList(p);
  for (unsigned i : valid_actions) {
    Action a = convert(i);
    float score;
    if (a.type == Action::kShift) {
      for (unsigned j = 0; j < subtype_dist.size(); ++j) {
        a.subtype = (WordId) j;
        score = log(type_dist[i]) + log(subtype_dist[j]);
        kbest.add(score, a);
      }
    }
    else {
      score = log(type_dist[i]);
      kbest.add(score, a);
    }
  }
  for (auto& thing : kbest.hypothesis_list()) {
    float score = get<0>(thing);
    Action a = get<1>(thing);
  }
  return kbest;
}

Expression AdhiParserBuilder::Loss(Expression state_vector, const Action& ref) const {
  assert (false);
  return Loss(state(), state_vector, ref);
}

Expression AdhiParserBuilder::Loss(RNNPointer p, Expression state_vector, const Action& ref) const {
  Expression action_dist = GetActionDistribution(p, state_vector);
  Expression neg_log_prob = -pick(action_dist, ref.GetIndex());
  if (ref.type == Action::kShift) {
    WordId wordid = ref.subtype;
    Expression loss = cfsm->neg_log_softmax(state_vector, wordid);
    neg_log_prob = neg_log_prob + loss;
  }
  return neg_log_prob;
}

Expression AdhiParserBuilder::BuildGraph(const vector<Action>& correct_actions) {
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

Expression AdhiParserBuilder::EmbedNonterminal(WordId nt, const vector<Expression>& children) {
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

bool AdhiParserBuilder::IsDone() const {
  return IsDone(state());
}

bool AdhiParserBuilder::IsDone(RNNPointer p) const {
  assert (p >= 0 && (unsigned)p < prev_states.size());
  const AdhiParserState& ps = prev_states[p];
  return ps.prev_action.type != Action::kNone && ps.nopen_parens == 0;
}

SourceConditionedAdhiParserBuilder::SourceConditionedAdhiParserBuilder() {}

SourceConditionedAdhiParserBuilder::SourceConditionedAdhiParserBuilder(Model& model, SoftmaxBuilder* cfsm,
    unsigned vocab_size, unsigned nt_vocab_size, unsigned action_vocab_size, unsigned hidden_dim,
    unsigned term_emb_dim, unsigned nt_emb_dim, unsigned source_dim) :
  AdhiParserBuilder(model, cfsm, vocab_size, nt_vocab_size, action_vocab_size, hidden_dim, term_emb_dim,
    nt_emb_dim),
  p_W(model.add_parameters({hidden_dim, source_dim})) {}

void SourceConditionedAdhiParserBuilder::NewGraph(ComputationGraph& cg) {
  AdhiParserBuilder::NewGraph(cg);
  W = parameter(cg, p_W);
}

Expression SourceConditionedAdhiParserBuilder::GetStateVector(Expression source_context) const {
  return GetStateVector(source_context, state());
}

Expression SourceConditionedAdhiParserBuilder::GetStateVector(Expression source_context, RNNPointer p) const {
  assert (p >= 0 && (unsigned)p < prev_states.size());
  AdhiParserState state = prev_states[p];
  Expression stack_summary = Summarize(stack_lstm, state.stack_lstm_pointer);

  Expression p_t = affine_transform({pbias, S, stack_summary, W, source_context});
  Expression nlp_t = rectify(p_t);
  return nlp_t;
}


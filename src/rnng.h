#pragma once
#include <vector>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "cnn/cfsm-builder.h"
#include "utils.h"

using namespace std;
using namespace cnn;

struct Action {
  enum ActionType {kNone, kShift, kNT, kReduce};
  ActionType type; // One of the above action types
  WordId subtype; // A word id (for shift) or non-terminal id (for nt). Otherwise unused.
  unsigned GetIndex() const;
};

class ParserBuilder;
struct ParserState {
  static const unsigned kMaxOpenNTs = 100;

  vector<Expression> stack; // variables representing subtree embeddings
  vector<Expression> terms; // generated terminals
  vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
  unsigned nopen_parens;
  unsigned nt_count; // number of times an NT has been introduced
  Action prev_action;
  unsigned action_count; // incremented at each prediction
  ParserBuilder* builder;

  explicit ParserState(ParserBuilder* builder);

  unsigned GetActionIndex(const Action& a);
  bool IsActionForbidden(const Action& a);
  vector<unsigned> GetValidActionList();
  void PerformShift(WordId wordid);
  void PerformNT(WordId ntid);
  void PerformReduce();
};

struct ParserBuilder {
friend class ParserState;
public:
  ParserBuilder();
  explicit ParserBuilder(Model& model, SoftmaxBuilder* cfsm, unsigned vocab_size, unsigned nt_vocab_size, unsigned action_vocab_size, unsigned hidden_dim, unsigned term_emb_dim, unsigned nt_emb_dim, unsigned action_emb_dim);
  void SetDropout(float rate);
  void NewGraph(ComputationGraph& cg);
  vector<Action> Sample(const vector<WordId>& sentence);
  vector<Action> Predict(const vector<WordId>& sentence);
  Expression Summarize(const LSTMBuilder& builder) const;
  Expression BuildGraph(const vector<Action>& correct_actions);

  void StartNewSentence(ParserState& state);
  Expression GetStateSummary(Expression stack_summary, Expression action_summary, Expression term_summary);
  Expression ComputeActionDistribution(Expression state_summary, const vector<unsigned>& valid_actions);
  Expression EmbedNonterminal(WordId nt, const vector<Expression>& children);

private:
  ComputationGraph* pcg;

  LSTMBuilder stack_lstm; // Stack
  LSTMBuilder term_lstm; // Sequence of generated terminals
  LSTMBuilder action_lstm; // Generated action sequence
  LSTMBuilder const_lstm_fwd; // Used to compose children of a node into a representation of the node 
  LSTMBuilder const_lstm_rev; // Used to compose children of a node into a representation of the node

  LookupParameter p_w; // word embeddings
  LookupParameter p_nt; // nonterminal embeddings
  LookupParameter p_ntup; // nonterminal embeddings when used in a composed representation
  LookupParameter p_a; // input action embeddings

  Parameter p_pbias; // parser state bias
  Parameter p_A; // action lstm to parser state
  Parameter p_S; // stack lstm to parser state
  Parameter p_T; // term lstm to parser state
  Parameter p_cW; // composition function weights
  Parameter p_cbias; // composition function bias
  Parameter p_p2a;   // parser state to action
  Parameter p_action_start;  // action bias
  Parameter p_abias;  // action bias
  Parameter p_stack_guard;  // end of stack

  Expression pbias;
  Expression A;
  Expression S;
  Expression T;
  Expression cW;
  Expression cbias;
  Expression p2a;
  Expression action_start;
  Expression abias;
  Expression stack_guard;

  SoftmaxBuilder* cfsm;
  float dropout_rate;
  unsigned nt_vocab_size;
};

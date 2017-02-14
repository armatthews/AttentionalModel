#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "dynet/dynet.h"
#include "dynet/lstm.h"
#include "dynet/cfsm-builder.h"
#include "kbestlist.h"
#include "utils.h"

using namespace std;
using namespace dynet;

struct Action {
  enum ActionType {kNone, kShift, kNT, kReduce};
  ActionType type; // One of the above action types
  WordId subtype; // A word id (for shift) or non-terminal id (for nt). Otherwise unused.
  unsigned GetIndex() const;

  bool operator==(const Action& o) const;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & type;
    ar & subtype;
  }
};

struct ActionHash : public unary_function<Action, size_t> {
  size_t operator()(const Action& action) const {
    return hash<unsigned>()(action.GetIndex());
  }
};

struct ParserState {
  static const unsigned kMaxOpenNTs = 100;

  vector<Expression> stack; // variables representing subtree embeddings
  vector<Expression> terms; // generated terminals
  vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
  unsigned nopen_parens;
  Action prev_action;

  RNNPointer stack_lstm_pointer;
  RNNPointer terminal_lstm_pointer;
  RNNPointer action_lstm_pointer;

  ParserState();

  bool IsActionForbidden(Action::ActionType at) const;
};

Action convert(unsigned id);

struct ParserBuilder {
public:
  ParserBuilder();
  explicit ParserBuilder(Model& model, SoftmaxBuilder* cfsm, unsigned vocab_size, unsigned nt_vocab_size, unsigned hidden_dim, unsigned term_emb_dim, unsigned nt_emb_dim, unsigned source_dim);
  virtual void SetDropout(float rate);
  virtual void NewGraph(ComputationGraph& cg);
  virtual void NewSentence();

  Expression Summarize(const LSTMBuilder& builder) const;
  Expression Summarize(const LSTMBuilder& builder, RNNPointer p) const;
  Expression GetInitialContext() const;
  Expression GetStateVector(Expression source_context) const;
  virtual Expression GetStateVector(Expression source_context, RNNPointer p) const;

  Expression GetActionDistribution(Expression state_vector) const;
  Expression Loss(Expression state_vector, const Action& ref) const;
  Action Sample(Expression state_pointer) const;
  KBestList<Action> PredictKBest(Expression state_vector, unsigned K) const;

  Expression GetActionDistribution(RNNPointer p, Expression state_vector) const;
  Expression Loss(RNNPointer p, Expression state_vector, const Action& ref) const;
  Action Sample(RNNPointer p, Expression state_pointer) const;
  KBestList<Action> PredictKBest(RNNPointer p, Expression state_vector, unsigned K) const;

  RNNPointer state() const;

  void PerformAction(const Action& action);
  void PerformAction(const Action& action, RNNPointer p);
  vector<unsigned> GetValidActionList() const;
  vector<unsigned> GetValidActionList(RNNPointer p) const;

  //Expression BuildGraph(const vector<Action>& correct_actions);

  Expression EmbedNonterminal(WordId nt, const vector<Expression>& children);
  bool IsDone() const;
  bool IsDone(RNNPointer p) const;

protected:
  ComputationGraph* pcg;
  ParserState* curr_state;
  vector<ParserState> prev_states;

  LSTMBuilder stack_lstm; // Stack
  LSTMBuilder const_lstm_fwd; // Used to compose children of a node into a representation of the node
  LSTMBuilder const_lstm_rev; // Used to compose children of a node into a representation of the node

  LookupParameter p_w; // word embeddings
  LookupParameter p_nt; // nonterminal embeddings
  LookupParameter p_ntup; // nonterminal embeddings when used in a composed representation

  Parameter p_pbias; // parser state bias
  Parameter p_S; // stack lstm to parser state
  Parameter p_W; // source context to parser state
  Parameter p_cW; // composition function weights
  Parameter p_cbias; // composition function bias
  Parameter p_p2a;   // parser state to action
  Parameter p_abias;  // action bias
  Parameter p_stack_guard;  // end of stack

  Expression pbias;
  Expression S;
  Expression W;
  Expression cW;
  Expression cbias;
  Expression p2a;
  Expression abias;
  Expression stack_guard;

  SoftmaxBuilder* cfsm;
  float dropout_rate;
  unsigned nt_vocab_size;

  virtual void PerformAction(const Action& action, const ParserState& state);
  virtual void PerformShift(WordId wordid);
  virtual void PerformNT(WordId ntid);
  virtual void PerformReduce();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & stack_lstm;
    ar & const_lstm_fwd;
    ar & const_lstm_rev;

    ar & p_w;
    ar & p_nt;
    ar & p_ntup;

    ar & p_pbias;
    ar & p_W;
    ar & p_S;
    ar & p_cW;
    ar & p_cbias;
    ar & p_p2a;
    ar & p_abias;
    ar & p_stack_guard;

    ar & cfsm;
    ar & nt_vocab_size;
  }
};

struct FullParserBuilder : public ParserBuilder {
public:
  FullParserBuilder();
  FullParserBuilder(Model& model, SoftmaxBuilder* cfsm,
    unsigned vocab_size, unsigned nt_vocab_size, unsigned hidden_dim,
    unsigned term_emb_dim, unsigned nt_emb_dim, unsigned action_emb_dim, unsigned source_dim);
  void SetDropout(float rate) override;
  Expression GetStateVector(Expression source_context, RNNPointer p) const override;
  void NewGraph(ComputationGraph& cg) override;
  void NewSentence() override;

protected:
  void PerformAction(const Action& action, const ParserState& state) override;
  void PerformShift(WordId wordid) override;
  void PerformNT(WordId ntid) override;
  void PerformReduce() override;

private:
  LSTMBuilder term_lstm; // Sequence of generated terminals
  LSTMBuilder action_lstm; // Generated action sequence
  LookupParameter p_a; // input action embeddings
  Parameter p_action_start;  // action LSTM start symbol
  Parameter p_A; // action lstm to parser state
  Parameter p_T; // term lstm to parser state

  Expression action_start;
  Expression A;
  Expression T;
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<ParserBuilder>(*this);
    ar & term_lstm;
    ar & action_lstm;
    ar & p_action_start;
    ar & p_a;
    ar & p_A;
    ar & p_T;
  }
};

BOOST_CLASS_EXPORT_KEY(FullParserBuilder)

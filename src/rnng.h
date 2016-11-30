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

  explicit ParserState();

  bool IsActionForbidden(const Action& a) const;
};

struct ParserBuilder {
friend class ParserState;
public:
  ParserBuilder();
  explicit ParserBuilder(Model& model, SoftmaxBuilder* cfsm, unsigned vocab_size, unsigned nt_vocab_size, unsigned action_vocab_size, unsigned hidden_dim, unsigned term_emb_dim, unsigned nt_emb_dim, unsigned action_emb_dim);
  virtual void SetDropout(float rate);
  virtual void NewGraph(ComputationGraph& cg);
  virtual void NewSentence();

  virtual Expression Summarize(const LSTMBuilder& builder) const;
  virtual Expression Summarize(const LSTMBuilder& builder, RNNPointer p) const;
  virtual Expression GetStateVector() const;
  virtual Expression GetStateVector(RNNPointer p) const;

  virtual Expression GetActionDistribution(Expression state_vector) const;
  virtual Expression Loss(Expression state_vector, const Action& ref) const;
  virtual Action Sample(Expression state_pointer) const;
  virtual KBestList<Action> PredictKBest(Expression state_vector, unsigned K) const;

  virtual Expression GetActionDistribution(RNNPointer p, Expression state_vector) const;
  virtual Expression Loss(RNNPointer p, Expression state_vector, const Action& ref) const;
  virtual Action Sample(RNNPointer p, Expression state_pointer) const;
  virtual KBestList<Action> PredictKBest(RNNPointer p, Expression state_vector, unsigned K) const;

  virtual RNNPointer state() const;

  virtual void PerformAction(const Action& action);
  virtual void PerformAction(const Action& action, RNNPointer p);
  virtual vector<unsigned> GetValidActionList() const;
  virtual vector<unsigned> GetValidActionList(RNNPointer p) const;

  virtual Expression BuildGraph(const vector<Action>& correct_actions);

  virtual Expression EmbedNonterminal(WordId nt, const vector<Expression>& children);
  virtual bool IsDone() const;
  virtual bool IsDone(RNNPointer p) const;

protected:
  ComputationGraph* pcg;
  ParserState* curr_state;
  vector<ParserState> prev_states;

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
  Parameter p_action_start;  // action LSTM start symbol
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

  void PerformAction(const Action& action, const ParserState& state);
  void PerformShift(WordId wordid);
  void PerformNT(WordId ntid);
  void PerformReduce();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & stack_lstm;
    ar & term_lstm;
    ar & action_lstm;
    ar & const_lstm_fwd;
    ar & const_lstm_rev;

    ar & p_w;
    ar & p_nt;
    ar & p_ntup;
    ar & p_a;

    ar & p_pbias;
    ar & p_A;
    ar & p_S;
    ar & p_T;
    ar & p_cW;
    ar & p_cbias;
    ar & p_p2a;
    ar & p_action_start;
    ar & p_abias;
    ar & p_stack_guard;

    ar & cfsm;
    ar & nt_vocab_size;
  }
};

struct SourceConditionedParserBuilder : public ParserBuilder {
  SourceConditionedParserBuilder();
  SourceConditionedParserBuilder(Model& model, SoftmaxBuilder* cfsm,
    unsigned vocab_size, unsigned nt_vocab_size, unsigned action_vocab_size, unsigned hidden_dim,
    unsigned term_emb_dim, unsigned nt_emb_dim, unsigned action_emb_dim, unsigned source_dim);

  void NewGraph(ComputationGraph& cg);
  Expression GetStateVector(Expression source_context) const;
  Expression GetStateVector(Expression source_context, RNNPointer p) const;
private:
  Parameter p_W;
  Expression W;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<ParserBuilder>(*this);
    ar & p_W;
  }
};
BOOST_CLASS_EXPORT_KEY(SourceConditionedParserBuilder)

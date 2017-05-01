#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "dynet/dynet.h"
#include "dynet/lstm.h"
#include "dynet/expr.h"
#include "dynet/cfsm-builder.h"
#include "mlp.h"
#include "embedder.h"
#include "utils.h"
#include "kbestlist.h"
#include "rnng.h"

using namespace std;
using namespace dynet;
using namespace dynet::expr;

class OutputModel {
public:
  virtual ~OutputModel();

  virtual void NewGraph(ComputationGraph& cg) = 0;
  virtual void SetDropout(float rate) {}
  virtual Expression GetState() const;
  virtual Expression GetState(RNNPointer p) const = 0;
  virtual RNNPointer GetStatePointer() const = 0;
  virtual Expression AddInput(const shared_ptr<const Word> prev_word, const Expression& context);
  virtual Expression AddInput(const shared_ptr<const Word> prev_word, const Expression& context, const RNNPointer& p) = 0;

  virtual Expression PredictLogDistribution(Expression context);
  virtual Expression PredictLogDistribution(RNNPointer p, Expression context) = 0;
  virtual KBestList<shared_ptr<Word>> PredictKBest(Expression context, unsigned K);
  virtual KBestList<shared_ptr<Word>> PredictKBest(RNNPointer p, Expression context, unsigned K) = 0;
  virtual pair<shared_ptr<Word>, float> Sample(Expression context);
  virtual pair<shared_ptr<Word>, float> Sample(RNNPointer p, Expression context) = 0;
  virtual Expression Loss(Expression context, const shared_ptr<const Word> ref);
  virtual Expression Loss(RNNPointer p, Expression context, const shared_ptr<const Word> ref) = 0;

  virtual bool IsDone() const;
  virtual bool IsDone(RNNPointer p) const = 0;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class SoftmaxOutputModel : public OutputModel {
public:
  SoftmaxOutputModel();
  SoftmaxOutputModel(Model& model, unsigned embedding_dim, unsigned context_dim, unsigned state_dim, Dict* vocab, const string& clusters_file);

  void NewGraph(ComputationGraph& cg) override;
  void SetDropout(float rate) override;
  virtual Expression GetState(RNNPointer p) const override;
  RNNPointer GetStatePointer() const override;
  Expression AddInput(const shared_ptr<const Word> prev_word, const Expression& context, const RNNPointer& p) override;

  virtual Expression PredictLogDistribution(RNNPointer p, Expression context) override;
  virtual KBestList<shared_ptr<Word>> PredictKBest(RNNPointer p, Expression context, unsigned K) override;
  virtual pair<shared_ptr<Word>, float> Sample(RNNPointer p, Expression context) override;
  virtual Expression Loss(RNNPointer p, Expression context, const shared_ptr<const Word> ref) override;

  bool IsDone(RNNPointer p) const override;

  // TODO: Take an (standard?) embedder
  Expression Embed(const shared_ptr<const StandardWord> word);

//protected:
  WordId kEOS;
  unsigned state_dim;
  LSTMBuilder output_builder;
  Parameter p_output_builder_initial_state;
  LookupParameter embeddings;
  SoftmaxBuilder* fsb;

  vector<bool> done;
  Expression output_builder_initial_state;
  ComputationGraph* pcg;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<OutputModel>(*this);
    ar & kEOS;
    ar & state_dim;
    ar & output_builder;
    ar & p_output_builder_initial_state;
    ar & embeddings;
    ar & fsb;
  }
};
BOOST_CLASS_EXPORT_KEY(SoftmaxOutputModel)

class MlpSoftmaxOutputModel : public SoftmaxOutputModel {
public:
  MlpSoftmaxOutputModel();
  MlpSoftmaxOutputModel(Model& model, unsigned embedding_dim, unsigned context_dim, unsigned state_dim, unsigned hidden_dim, Dict* vocab, const string& clusters_file);

  Expression GetState(RNNPointer p) const override;
  Expression AddInput(const shared_ptr<const Word> prev_word, const Expression& context, const RNNPointer& p) override;

  void NewGraph(ComputationGraph& cg) override;

  // TODO: What are these and why are the necessary? I think maybe for the continuous decoder?
  virtual Expression AddInput(Expression prev_word_emb, const Expression& context);
  virtual Expression AddInput(Expression prev_word_emb, const Expression& context, const RNNPointer& p);
private:
  Parameter p_W, p_b;
  Expression W, b;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<SoftmaxOutputModel>(*this);
    ar & p_W;
    ar & p_b;
  }
};
BOOST_CLASS_EXPORT_KEY(MlpSoftmaxOutputModel)

class MorphologyOutputModel : public OutputModel {
public:
  MorphologyOutputModel();
  MorphologyOutputModel(Model& model, Dict& word_vocab, Dict& root_vocab, unsigned affix_vocab_size, unsigned char_vocab_size, unsigned word_emb_dim, unsigned root_emb_dim, unsigned affix_emb_dim, unsigned char_emb_dim, unsigned model_chooser_hidden_dim, unsigned affix_init_hidden_dim, unsigned char_init_hidden_dim, unsigned state_dim, unsigned affix_lstm_dim, unsigned char_lstm_dim, unsigned context_dim, const string& word_clusters, const string& root_clusters);

  void NewGraph(ComputationGraph& cg) override;
  void SetDropout(float rate) override;
  Expression GetState(RNNPointer p) const override;
  RNNPointer GetStatePointer() const override;
  Expression AddInput(const shared_ptr<const Word> prev_word, const Expression& context, const RNNPointer& p) override;

  Expression PredictLogDistribution(RNNPointer p, Expression context) override;
  KBestList<shared_ptr<Word>> PredictKBest(RNNPointer p, Expression context, unsigned K) override;
  pair<shared_ptr<Word>, float> Sample(RNNPointer p, Expression context) override;
  Expression Loss(RNNPointer p, Expression context, const shared_ptr<const Word> ref) override;

  bool IsDone(RNNPointer p) const override;

  Expression WordLoss(const Expression& state, const WordId ref);
  Expression AnalysisLoss(const Expression& state, const Analysis& ref);
  Expression MorphLoss(const Expression& state, const vector<Analysis>& ref);
  Expression CharLoss(const Expression& state, const vector<WordId>& ref);

private:
  unsigned state_dim;
  unsigned affix_lstm_dim;
  unsigned char_lstm_dim;

  MLP model_chooser;
  MLP affix_lstm_init;
  MLP char_lstm_init;
  LSTMBuilder affix_lstm;
  LSTMBuilder char_lstm;
  LSTMBuilder output_builder;
  Parameter output_lstm_init;

  MorphologyEmbedder embedder;
  LookupParameter root_embeddings;
  LookupParameter affix_embeddings;
  LookupParameter char_embeddings;
  SoftmaxBuilder* word_softmax;
  SoftmaxBuilder* root_softmax;
  SoftmaxBuilder* affix_softmax;
  SoftmaxBuilder* char_softmax;

  vector<Expression> output_lstm_init_v;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<OutputModel>(*this);
    ar & state_dim & affix_lstm_dim & char_lstm_dim;

    ar & model_chooser;
    ar & affix_lstm_init;
    ar & char_lstm_init;
    ar & affix_lstm;
    ar & char_lstm;
    ar & output_builder;
    ar & output_lstm_init;

    ar & embedder;
    ar & root_embeddings;
    ar & affix_embeddings;
    ar & char_embeddings;
    ar & word_softmax;
    ar & root_softmax;
    ar & affix_softmax;
    ar & char_softmax;
  }
};
BOOST_CLASS_EXPORT_KEY(MorphologyOutputModel)

class RnngOutputModel : public OutputModel {
public:
  RnngOutputModel();
  RnngOutputModel(Model& model, unsigned term_emb_dim, unsigned nt_emb_dim, unsigned action_emb_dim, unsigned source_dim, unsigned hidden_dim, Dict* vocab, const string& clusters_file);
  void InitializeDictionaries(const Dict& raw_vocab);

  void NewGraph(ComputationGraph& cg) override;
  void SetDropout(float rate) override;
  Expression GetState(RNNPointer p) const override;
  RNNPointer GetStatePointer() const override;
  Expression AddInput(const shared_ptr<const Word> prev_word, const Expression& context, const RNNPointer& p) override;

  Expression PredictLogDistribution(RNNPointer p, Expression context) override;
  KBestList<shared_ptr<Word>> PredictKBest(RNNPointer p, Expression context, unsigned K) override;
  pair<shared_ptr<Word>, float> Sample(RNNPointer p, Expression context) override;
  Expression Loss(RNNPointer p, Expression context, const shared_ptr<const Word> ref) override;

  bool IsDone(RNNPointer p) const override;

private:
  Action Convert(const WordId w) const;
  WordId Convert(const Action& a) const;

  ParserBuilder* builder;
  unsigned hidden_dim;

  vector<Action> w2a;
  unordered_map<Action, WordId, ActionHash> a2w;
  Dict raw_vocab, term_vocab, nt_vocab;

  vector<Expression> source_contexts;
  vector<Expression> state_context_vectors;
  vector<vector<StandardWord>> word_sequences;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<OutputModel>(*this);
    ar & builder;
    ar & hidden_dim;
    ar & w2a & a2w;
    ar & raw_vocab & term_vocab & nt_vocab;
  }
};
BOOST_CLASS_EXPORT_KEY(RnngOutputModel)

class DependencyOutputModel : public OutputModel {
public:
  DependencyOutputModel();
  DependencyOutputModel(Model& model, Embedder* embedder, unsigned context_dim, unsigned state_dim, unsigned final_hidden_dim, Dict& vocab);

  void NewGraph(ComputationGraph& cg) override;
  void SetDropout(float rate) override;
  Expression GetState(RNNPointer p) const override;
  RNNPointer GetStatePointer() const override;
  Expression AddInput(const shared_ptr<const Word> prev_word, const Expression& context, const RNNPointer& p) override;

  Expression PredictLogDistribution(RNNPointer p, Expression context) override;
  KBestList<shared_ptr<Word>> PredictKBest(RNNPointer p, Expression context, unsigned K) override;
  pair<shared_ptr<Word>, float> Sample(RNNPointer p, Expression context) override;
  Expression Loss(RNNPointer p, Expression context, const shared_ptr<const Word> ref) override;
  bool IsDone(RNNPointer p) const override;

private:
  typedef tuple<RNNPointer, RNNPointer, unsigned, bool> State; // Stack pointer, comp pointer, stack depth, done with left

  Embedder* embedder;
  LSTMBuilder stack_lstm;
  LSTMBuilder comp_lstm;
  MLP final_mlp;

  Parameter emb_transform_p; // Simple linear transform from word embedding space to state space
  Parameter stack_lstm_init_p;
  Parameter comp_lstm_init_p;

  Expression emb_transform;
  vector<Expression> stack_lstm_init;
  vector<Expression> comp_lstm_init;

  unsigned half_state_dim;
  unsigned done_with_left;
  unsigned done_with_right;

  vector<State> prev_states;
  vector<RNNPointer> stack; // From each state, if you were to see </RIGHT> where would you go back to?
  vector<RNNPointer> head;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<OutputModel>(*this);
    ar & embedder;
    ar & stack_lstm;
    ar & comp_lstm;
    ar & final_mlp;

    ar & emb_transform_p;
    ar & stack_lstm_init_p;
    ar & comp_lstm_init_p;

    ar & half_state_dim;
    ar & done_with_left;
    ar & done_with_right;
  }
};
BOOST_CLASS_EXPORT_KEY(DependencyOutputModel)


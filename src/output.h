#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unordered_map.hpp>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "cnn/expr.h"
#include "cnn/cfsm-builder.h"
#include "mlp.h"
#include "morphology.h"
#include "utils.h"
#include "rnng.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class OutputModel {
public:
  virtual ~OutputModel();

  virtual void NewGraph(ComputationGraph& cg) = 0;
  virtual void SetDropout(float rate) {}
  virtual Expression GetState() const;
  virtual Expression GetState(RNNPointer p) const = 0;
  virtual RNNPointer GetStatePointer() const = 0;
  virtual Expression AddInput(const Word* const prev_word, const Expression& context) = 0;
  virtual Expression AddInput(const Word* const prev_word, const Expression& context, const RNNPointer& p) = 0;
  virtual Expression PredictLogDistribution(const Expression& state) = 0;
  virtual Word* Sample(const Expression& state) = 0;
  virtual Expression Loss(const Expression& state, const Word* const ref) = 0;

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

  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  Expression GetState(RNNPointer p) const;
  RNNPointer GetStatePointer() const;
  Expression Embed(const StandardWord* word);
  Expression AddInput(const Word* const prev_word, const Expression& context);
  Expression AddInput(const Word* const prev_word, const Expression& context, const RNNPointer& p);
  Expression PredictLogDistribution(const Expression& state);
  Word* Sample(const Expression& state);
  Expression Loss(const Expression& state, const Word* const ref);

  bool IsDone(RNNPointer p) const;
  WordId kEOS;
protected:
  vector<bool> done;
  unsigned state_dim;
  LSTMBuilder output_builder;
  Parameter p_output_builder_initial_state;
  Expression output_builder_initial_state;
  LookupParameter embeddings;
  SoftmaxBuilder* fsb;
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
    //Parameter& prev = output_builder.params.back().back(); // XXX: Super hacky
    //p_output_builder_initial_state = Parameter(prev.mp, prev.index + 1);
    ar & embeddings;
    ar & fsb;
  }
};
BOOST_CLASS_EXPORT_KEY(SoftmaxOutputModel)

class MlpSoftmaxOutputModel : public SoftmaxOutputModel {
public:
  MlpSoftmaxOutputModel();
  MlpSoftmaxOutputModel(Model& model, unsigned embedding_dim, unsigned context_dim, unsigned state_dim, unsigned hidden_dim, Dict* vocab, const string& clusters_file);

  void NewGraph(ComputationGraph& cg);
  Expression PredictLogDistribution(const Expression& state);
  Word* Sample(const Expression& state);
  Expression Loss(const Expression& state, const Word* const ref);
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
  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  Expression GetState(RNNPointer p) const;
  RNNPointer GetStatePointer() const;
  Expression AddInput(const Word* const prev_word, const Expression& context);
  Expression AddInput(const Word* const prev_word, const Expression& context, const RNNPointer& p);
  Expression PredictLogDistribution(const Expression& state);
  Word* Sample(const Expression& state);

  Expression WordLoss(const Expression& state, const WordId ref);
  Expression AnalysisLoss(const Expression& state, const Analysis& ref);
  Expression MorphLoss(const Expression& state, const vector<Analysis>& ref);
  Expression CharLoss(const Expression& state, const vector<WordId>& ref);
  Expression Loss(const Expression& state, const Word* const ref);

  bool IsDone(RNNPointer p) const;

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
  vector<Expression> output_lstm_init_v;
  MorphologyEmbedder embedder;
  LookupParameter root_embeddings;
  LookupParameter affix_embeddings;
  LookupParameter char_embeddings;
  SoftmaxBuilder* word_softmax;
  SoftmaxBuilder* root_softmax;
  SoftmaxBuilder* affix_softmax;
  SoftmaxBuilder* char_softmax;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<OutputModel>(*this);
    ar & state_dim & affix_lstm_dim & char_lstm_dim;
    ar & model_chooser;
    ar & affix_lstm_init;
    ar & char_lstm_init;
    ar & output_builder;
    ar & embedder;
    ar & root_embeddings;
    ar & char_embeddings;
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
  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  Expression GetState() const;
  Expression GetState(RNNPointer p) const;
  RNNPointer GetStatePointer() const;
  Expression AddInput(const Word* const prev_word, const Expression& context);
  Expression AddInput(const Word* const prev_word, const Expression& context, const RNNPointer& p);
  Expression PredictLogDistribution(const Expression& source_context);
  Word* Sample(const Expression& source_context);
  Expression Loss(const Expression& source_context, const Word* const ref);

  bool IsDone(RNNPointer p) const;

private:
  SourceConditionedParserBuilder* builder;
  Action Convert(const WordId w) const;
  WordId Convert(const Action& a) const;
  Expression most_recent_source_thing; // XXX
  unsigned hidden_dim;
  ComputationGraph* pcg;

  vector<Action> w2a;
  unordered_map<Action, WordId, ActionHash> a2w;
  Dict raw_vocab, term_vocab, nt_vocab;

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

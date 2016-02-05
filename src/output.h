#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "cnn/expr.h"
#include "cnn/cfsm-builder.h"
#include "utils.h"
#include "kbestlist.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class OutputModel {
public:
  virtual ~OutputModel();

  virtual void NewGraph(ComputationGraph& cg) = 0;
  virtual Expression GetState() = 0;
  //virtual KBestList<WordId> Predict(const unsigned K, const WordId prev_word, const Expression& context) = 0;
  virtual WordId Sample(const WordId prev_word, const Expression& context) = 0;
  virtual Expression Loss(const WordId prev_word, const Expression& context, unsigned ref) = 0;

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
  Expression GetState();
  WordId Sample(const WordId prev_word, const Expression& context);
  Expression Loss(const WordId prev_word, const Expression& context, unsigned ref);
private:
  unsigned embedding_dim, context_dim, state_dim, vocab_size;
  LSTMBuilder output_builder;
  LookupParameterIndex embeddings;
  SoftmaxBuilder* fsb;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<SoftmaxOutputModel, OutputModel>();
    ar & embedding_dim;
    ar & context_dim;
    ar & state_dim;
    ar & vocab_size;
    ar & output_builder;
    ar & embeddings;
    ar & fsb;
  }
};
BOOST_CLASS_EXPORT_KEY(SoftmaxOutputModel)

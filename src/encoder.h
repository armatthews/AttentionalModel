#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "cnn/expr.h"
#include "utils.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

template <class Input>
class EncoderModel {
public:
  virtual ~EncoderModel() {}

  virtual void InitializeParameters(Model& model) = 0;
  virtual void NewGraph(ComputationGraph& cg) = 0;
  virtual vector<Expression> Encode(const Input& sentence) = 0;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    cerr << "serializing encodermodel" << endl;
  }
};

class BidirectionalSentenceEncoder : public EncoderModel<Sentence> {
public:
  BidirectionalSentenceEncoder();
  BidirectionalSentenceEncoder(Model& model, unsigned vocab_size, unsigned input_dim, unsigned output_dim);
  void InitializeParameters(Model& model);

  void NewGraph(ComputationGraph& cg);
  vector<Expression> Encode(const Sentence& sentence);
  vector<Expression> EncodeForward(const Sentence& sentence);
  vector<Expression> EncodeReverse(const Sentence& sentence);
private:
  unsigned vocab_size;
  unsigned input_dim;
  unsigned output_dim;
  LSTMBuilder forward_builder;
  LSTMBuilder reverse_builder;
  LookupParameters* embeddings;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<BidirectionalSentenceEncoder, EncoderModel>();
    cerr << "serializing bidirectionalsentenceencoder" << endl;
    ar & vocab_size;
    ar & input_dim;
    ar & output_dim;
  }
};
BOOST_CLASS_EXPORT_KEY(BidirectionalSentenceEncoder)

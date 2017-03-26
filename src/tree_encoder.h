#pragma once
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "encoder.h"
//#include "treelstm.h"
#include "dynet/treelstm.h"
#include "syntax_tree.h"

class TreeEncoder : public EncoderModel {
public:
  TreeEncoder();
  TreeEncoder(Model& model, unsigned vocab_size, unsigned label_vocab_size, unsigned input_dim, unsigned output_dim);

  void NewGraph(ComputationGraph& cg);
  vector<Expression> Encode(const InputSentence* const input);
  Expression EncodeSentence(const InputSentence* const input);
private:
  unsigned output_dim;
  EncoderModel* linear_encoder;
  dynet::TreeLSTMBuilder* tree_builder;
  LookupParameter label_embeddings;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<EncoderModel>(*this);
    ar & output_dim;
    ar & linear_encoder;
    ar & tree_builder;
    ar & label_embeddings;
  }
};
BOOST_CLASS_EXPORT_KEY(TreeEncoder)

#pragma once
#include "encoder.h"
#include "treelstm.h"
#include "syntax_tree.h"

class TreeEncoder : public EncoderModel {
public:
  TreeEncoder();
  TreeEncoder(Model& model, unsigned vocab_size, unsigned label_vocab_size, unsigned input_dim, unsigned output_dim);
  void InitializeParameters(Model& model);

  bool IsT2S() const;
  void NewGraph(ComputationGraph& cg);
  vector<Expression> Encode(const TranslatorInput* const input);
private:
  unsigned vocab_size;
  unsigned label_vocab_size;
  unsigned input_dim;
  unsigned output_dim;
  EncoderModel* linear_encoder;
  TreeLSTMBuilder* tree_builder;
  LookupParameter label_embeddings;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    //boost::serialization::void_cast_register<TreeEncoder, EncoderModel>();
    ar & boost::serialization::base_object<EncoderModel>(*this);
  }
};
BOOST_CLASS_EXPORT_KEY(TreeEncoder)

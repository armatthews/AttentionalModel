#pragma once
#include "encoder.h"
#include "treelstm.h"
#include "syntax_tree.h"

class TreeEncoder : public EncoderModel<SyntaxTree> {
public:
  TreeEncoder();
  TreeEncoder(Model& model, unsigned vocab_size, unsigned label_vocab_size, unsigned input_dim, unsigned output_dim);
  void InitializeParameters(Model& model);

  void NewGraph(ComputationGraph& cg);
  vector<Expression> Encode(const SyntaxTree& sentence);
private:
  unsigned vocab_size;
  unsigned label_vocab_size;
  unsigned input_dim;
  unsigned output_dim;
  EncoderModel<Sentence>* linear_encoder;
  TreeLSTMBuilder* tree_builder;
  LookupParameterIndex label_embeddings;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<TreeEncoder, EncoderModel>();
  }
};
BOOST_CLASS_EXPORT_KEY(TreeEncoder)

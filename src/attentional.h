#pragma once
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include "treelstm.h"
#include "bitext.h"
#include "kbestlist.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

struct OutputState {
  // State is the LSTM state
  Expression state;
  // Context is a weighted sum of annotation vectors
  Expression context;

  // rnn_pointer allows us to restore the state of output_builder
  RNNPointer rnn_pointer;
};

// A simple 2 layer MLP
struct MLP {
  vector<Expression> i_IH;
  Expression i_Hb;
  Expression i_HO;
  Expression i_Ob;

  Expression Feed(vector<Expression> input) const;
};

struct PartialHypothesis {
  vector<WordId> words;
  OutputState state;
};

class AttentionalModel {
  friend class AttentionalDecoder;
public:
  AttentionalModel(Model& model, unsigned src_vocab_size, unsigned tgt_vocab_size);
  Expression BuildGraph(const vector<WordId>& source, const vector<WordId>& target, ComputationGraph& hg);
  Expression BuildGraph(const SyntaxTree& source, const vector<WordId>& target, ComputationGraph& hg);

protected:
  vector<Expression> BuildForwardAnnotations(const vector<WordId>& sentence, ComputationGraph& hg);
  vector<Expression> BuildReverseAnnotations(const vector<WordId>& sentence, ComputationGraph& hg);
  vector<Expression> BuildAnnotationVectors(const vector<Expression>& forward_contexts, const vector<Expression>& reverse_contexts, ComputationGraph& hg);
  vector<Expression> BuildTreeAnnotationVectors(const SyntaxTree& source_tree, const vector<Expression>& linear_annotations, ComputationGraph& cg);
  tuple<vector<Expression>, Expression> BuildAnnotationVectors(const vector<WordId>& source, ComputationGraph& cg);
  tuple<vector<Expression>, Expression> BuildAnnotationVectors(const SyntaxTree& source_tree, ComputationGraph& cg);
  Expression BuildGraphGivenAnnotations(const vector<Expression>& annotations, Expression zeroth_context, const vector<WordId>& target, ComputationGraph& cg);
  OutputState GetNextOutputState(unsigned t, const Expression& context, const Expression& prev_target_word_embedding, const vector<Expression>& annotations, const MLP& aligner, ComputationGraph& hg, vector<float>* out_alignment = NULL);
  OutputState GetNextOutputState(unsigned t, const RNNPointer& rnn_pointer, const Expression& context, const Expression& prev_target_word_embedding, const vector<Expression>& annotations, const MLP& aligner, ComputationGraph& hg, vector<float>* out_alignment = NULL);
  Expression ComputeOutputDistribution(unsigned source_length, unsigned t, const WordId prev_word, const Expression state, const Expression context, const MLP& final_mlp, ComputationGraph& hg);
  vector<unsigned&> GetParams();
  MLP GetAligner(ComputationGraph& cg) const;
  MLP GetFinalMLP(ComputationGraph& cg) const;
  Expression GetZerothContext(Expression zeroth_reverse_annotation, ComputationGraph& cg) const;
  OutputState GetInitialOutputState(Expression zeroth_context, const vector<Expression>& annotations, const MLP& aligner, const WordId kSOS, ComputationGraph& cg, vector<float>* alignment = NULL);

private:
  LSTMBuilder forward_builder, reverse_builder, output_builder;
  TreeLSTMBuilder tree_builder;
  LookupParameters* p_Es; // source language word embedding matrix
  LookupParameters* p_Et; // target language word embedding matrix

  // Alignment NN
  Parameters* p_aIH1; // weight matrix between the input state and hidden layer
  Parameters* p_aIH2; // weight matrix between the input annotation and hidden layer
  Parameters* p_aHb; // Alignment NN hidden layer bias
  Parameters* p_aHO; // Alignment NN weight matrix between the hidden and output layers
  Parameters* p_aOb; // Alignment NN output layer bias;
  Parameters* p_Ws; // Used to compute p_0 from h_backwards_0
  Parameters* p_bs; // Used to compute p_0 from h_backwars_0

  // "Final" NN (from the tuple (y_{i-1}, s_i, c_i) to the distribution over output words y_i)
  Parameters* p_fIH1; // input prev embedding->hidden weights
  Parameters* p_fIH2; // input state->hidden weights
  Parameters* p_fIH3; // input context->hidden weights
  Parameters* p_fHb; // Same, hidden bias
  Parameters* p_fHO; // Same, hidden->output weights
  Parameters* p_fOb; // Same, output bias

  Parameters* p_tension; // diagonal tension for prior on alignments
  Parameters* p_length_multiplier; // |t| \approx this times |s|
  vector<cnn::real> zero_annotation; // Just a vector of zeros, the same size as an annotation vector
  vector<cnn::real> eos_onehot; // A one-hot vector the size of the vocabulary, with a 1 in the spot of </s>

  unsigned lstm_layer_count = 2;
  unsigned embedding_dim = 64; // Dimensionality of both source and target word embeddings. For now these are the same.
  unsigned half_annotation_dim = 64; // Dimensionality of h_forward and h_backward. The full h has twice this dimension.
  unsigned output_state_dim = 64; // Dimensionality of s_j, the state just before outputing target word y_j
  unsigned alignment_hidden_dim = 64; // Dimensionality of the hidden layer in the alignment FFNN
  unsigned final_hidden_dim = 64; // Dimensionality of the hidden layer in the "final" FFNN

  Expression alignment_prior(unsigned t, unsigned source_length, ComputationGraph& cg);

  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & lstm_layer_count;
    ar & embedding_dim;
    ar & half_annotation_dim;
    ar & output_state_dim;
    ar & alignment_hidden_dim;
    ar & final_hidden_dim;
  }
};

#include <queue>
#include <cmath>
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"

#include "bitext.h"
#include "attentional.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

Expression MLP::Feed(vector<Expression> inputs) const {
  assert (inputs.size() == i_IH.size());
  vector<Expression> xs(2 * inputs.size() + 1);
  xs[0] = i_Hb;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    xs[2 * i + 1] = i_IH[i];
    xs[2 * i + 2] = inputs[i];
  }
  Expression hidden1 = affine_transform(xs);
  Expression hidden2 = tanh({hidden1});
  Expression output = affine_transform({i_Ob, i_HO, hidden2});
  return output;
}

AttentionalModel::AttentionalModel() {
  p_Es = p_Et = nullptr;
  p_aIH1 = p_aIH2 = p_aHb = nullptr;
  p_aHO = p_aOb = nullptr;
  p_Ws = p_bs = p_Ls = nullptr;
  p_fIH1 = p_fIH2 = p_fIH3 = p_fHb = nullptr;
  p_fHO = p_fOb = nullptr;
  p_tension = p_length_multiplier = nullptr;
}

AttentionalModel::AttentionalModel(Model& model, unsigned src_vocab_size, unsigned tgt_vocab_size) {
  InitializeParameters(model, src_vocab_size, tgt_vocab_size);
}

void AttentionalModel::InitializeParameters(Model& model, unsigned src_vocab_size, unsigned tgt_vocab_size) {
  forward_builder = LSTMBuilder(lstm_layer_count, embedding_dim, half_annotation_dim, &model);
  reverse_builder = LSTMBuilder(lstm_layer_count, embedding_dim, half_annotation_dim, &model);
  output_builder = LSTMBuilder(lstm_layer_count, embedding_dim + 2 * half_annotation_dim, output_state_dim, &model);
  tree_builder = new SocherTreeLSTMBuilder(5, lstm_layer_count, 2 * half_annotation_dim, 2 * half_annotation_dim, &model);
  p_Es = model.add_lookup_parameters(src_vocab_size, {embedding_dim});
  p_Et = model.add_lookup_parameters(tgt_vocab_size, {embedding_dim});
  p_aIH1 = model.add_parameters({alignment_hidden_dim, output_state_dim});
  p_aIH2 = model.add_parameters({alignment_hidden_dim, 2 * half_annotation_dim});
  p_aHb = model.add_parameters({alignment_hidden_dim, 1});
  p_aHO = model.add_parameters({1, alignment_hidden_dim});
  p_aOb = model.add_parameters({1, 1});
  // The paper says s_0 = tanh(Ws * h1_reverse), and that Ws is an N x N matrix, but my calculations show that the dimensionality must be as below.
  p_Ws = model.add_parameters({output_state_dim, half_annotation_dim});
  p_bs = model.add_parameters({output_state_dim});
  p_Ls = model.add_parameters({2 * half_annotation_dim, embedding_dim});

  p_fIH1 = model.add_parameters({final_hidden_dim, embedding_dim});
  p_fIH2 = model.add_parameters({final_hidden_dim, output_state_dim});
  p_fIH3 = model.add_parameters({final_hidden_dim, 2 * half_annotation_dim});
  p_fHb = model.add_parameters({final_hidden_dim});
  p_fHO = model.add_parameters({tgt_vocab_size, final_hidden_dim});
  p_fOb = model.add_parameters({tgt_vocab_size});

  p_tension = model.add_parameters({1,1});
  p_length_multiplier = model.add_parameters({1,1});
}

Expression AttentionalModel::BuildGraph(const vector<WordId>& source, const vector<WordId>& target, ComputationGraph& cg) {
  assert (target.size() >= 2 && target[0] == 1 && target[target.size() - 1] == 2);
  vector<Expression> annotations;
  Expression zeroth_state;
  tie(annotations, zeroth_state) = BuildAnnotationVectors(source, cg);

  return BuildGraphGivenAnnotations(annotations, zeroth_state, target, cg);
}

Expression AttentionalModel::BuildGraph(const SyntaxTree& source_tree, const vector<WordId>& target, ComputationGraph& cg) {
  assert (target.size() >= 2 && target[0] == 1 && target[target.size() - 1] == 2);
  vector<Expression> annotations;
  Expression zeroth_state;
  tie(annotations, zeroth_state) = BuildAnnotationVectors(source_tree, cg);

  return BuildGraphGivenAnnotations(annotations, zeroth_state, target, cg);
}

vector<Expression> AttentionalModel::BuildForwardAnnotations(const vector<WordId>& sentence, ComputationGraph& cg) {
  forward_builder.new_graph(cg);
  forward_builder.start_new_sequence();
  vector<Expression> forward_annotations(sentence.size());
  for (unsigned t = 0; t < sentence.size(); ++t) {
    Expression i_x_t = lookup(cg, p_Es, sentence[t]);
    Expression i_y_t = forward_builder.add_input(i_x_t);
    forward_annotations[t] = i_y_t;
  }
  return forward_annotations;
}

vector<Expression> AttentionalModel::BuildReverseAnnotations(const vector<WordId>& sentence, ComputationGraph& cg) {
  reverse_builder.new_graph(cg);
  reverse_builder.start_new_sequence();
  vector<Expression> reverse_annotations(sentence.size());
  for (unsigned t = sentence.size(); t > 0; ) {
    t--;
    Expression i_x_t = lookup(cg, p_Es, sentence[t]);
    Expression i_y_t = reverse_builder.add_input(i_x_t);
    reverse_annotations[t] = i_y_t;
  }
  return reverse_annotations;
}

vector<Expression> AttentionalModel::BuildAnnotationVectors(const vector<Expression>& forward_annotations, const vector<Expression>& reverse_annotations, ComputationGraph& cg) {
  vector<Expression> annotations(forward_annotations.size());
  for (unsigned t = 0; t < forward_annotations.size(); ++t) {
    const Expression& i_f = forward_annotations[t];
    const Expression& i_r = reverse_annotations[t];
    Expression i_h = concatenate({i_f, i_r});
    annotations[t] = i_h;
  }
  return annotations;
}

vector<Expression> AttentionalModel::BuildTreeAnnotationVectors(const SyntaxTree& source_tree, const vector<Expression>& linear_annotations, ComputationGraph& cg) {
  tree_builder->new_graph(cg);
  tree_builder->start_new_sequence();
  vector<Expression> annotations;
  vector<Expression> tree_annotations;
  vector<const SyntaxTree*> node_stack = {&source_tree};
  vector<unsigned> index_stack = {0};
  unsigned terminal_index = 0;

  while (node_stack.size() > 0) {
    assert (node_stack.size() == index_stack.size());
    const SyntaxTree* node = node_stack.back();
    unsigned i = index_stack.back();
    if (i >= node->NumChildren()) {
      assert (tree_annotations.size() == node->id());
      vector<int> children(node->NumChildren());
      for (unsigned j = 0; j < node->NumChildren(); ++j) {
        unsigned child_id = node->GetChild(j).id();
        assert (child_id < tree_annotations.size());
        assert (child_id < (unsigned)INT_MAX);
        children[j] = (int)child_id;
      }

      Expression input_expr;
      if (node->NumChildren() == 0) {
        assert (terminal_index < linear_annotations.size());
        input_expr = linear_annotations[terminal_index];
        terminal_index++;
      }
      else {
        input_expr = parameter(cg, p_Ls) * lookup(cg, p_Es, node->label());
      }
      Expression node_annotation = tree_builder->add_input((int)node->id(), children, input_expr);
      tree_annotations.push_back(node_annotation);
      index_stack.pop_back();
      node_stack.pop_back();
    }
    else {
      index_stack[index_stack.size() - 1] += 1;
      node_stack.push_back(&node->GetChild(i));
      index_stack.push_back(0);
      ++i;
    }
  }
  assert (node_stack.size() == index_stack.size());

  return tree_annotations;
}

tuple<vector<Expression>, Expression> AttentionalModel::BuildAnnotationVectors(const vector<WordId>& source, ComputationGraph& cg) {
  vector<Expression> forward_annotations = BuildForwardAnnotations(source, cg);
  vector<Expression> reverse_annotations = BuildReverseAnnotations(source, cg);
  vector<Expression> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, cg);
  Expression zeroth_state = GetZerothState(reverse_annotations[0], cg);
  return make_tuple(annotations, zeroth_state);
}

tuple<vector<Expression>, Expression> AttentionalModel::BuildAnnotationVectors(const SyntaxTree& source_tree, ComputationGraph& cg) {
  vector<Expression> linear_annotations;
  Expression zeroth_state;
  vector<WordId> source = source_tree.GetTerminals();
  tie(linear_annotations, zeroth_state) = BuildAnnotationVectors(source, cg);
  vector<Expression> annotations = BuildTreeAnnotationVectors(source_tree, linear_annotations, cg);
  return make_tuple(annotations, zeroth_state);
}

Expression AttentionalModel::GetAlignments(unsigned t, Expression& prev_state,
    const vector<Expression>& annotations, const MLP& aligner,
    ComputationGraph& cg, vector<float>* out_alignment) {
  const unsigned source_size = annotations.size();

  Expression annotation_matrix = concatenate_cols(annotations);

  // The two loops below accomplish exactly the same thing.
  // The second one is slightly faster (at least for large-ish state sizes),
  // because W * prev_state does not change with respect to s, so we have
  // factored it out. The code below is kind of ugly, so we leave the simple
  // loop here for clarity.
  /*vector<Expression> unnormalized_alignments(source_size); // e_ij
  for (unsigned s = 0; s < source_size; ++s) {
    unnormalized_alignments[s] = aligner.Feed({prev_state, annotations[s]});
  }
  Expression unnormalized_alignment_vector = concatenate(unnormalized_alignments);// + log(alignment_prior(t, source_size, cg));*/

  // Implementation two: Slightly faster
  /*vector<Expression> unnormalized_alignments(source_size); // e_ij
  Expression new_bias = affine_transform({aligner.i_Hb, aligner.i_IH[0], prev_state});
  for (unsigned s = 0; s < source_size; ++s) {
    Expression hidden = tanh(affine_transform({new_bias, aligner.i_IH[1], annotations[s]}));
    Expression output = affine_transform({aligner.i_Ob, aligner.i_HO, hidden});
    unnormalized_alignments[s] = output;
  }
  Expression unnormalized_alignment_vector = concatenate(unnormalized_alignments);// + log(alignment_prior(t, source_size, cg));*/

  // Yet another implementation of the above, this time batching all the calls to the MLP into matrix-matrix operations
  // This yields a little over 10% speed up over the above.
  Expression new_bias = affine_transform({aligner.i_Hb, aligner.i_IH[0], prev_state});
  Expression bias_broadcast1 = concatenate_cols(vector<Expression>(annotations.size(), new_bias));
  Expression hidden = tanh(affine_transform({bias_broadcast1, aligner.i_IH[1], annotation_matrix}));
  Expression bias_broadcast2 = concatenate_cols(vector<Expression>(annotations.size(), aligner.i_Ob));
  Expression unnormalized_alignment_vector = transpose(affine_transform({bias_broadcast2, aligner.i_HO, hidden}));// + log(alignment_prior(t, source_size, cg));

  Expression normalized_alignment_vector = softmax(unnormalized_alignment_vector);
  if (out_alignment != NULL) {
    *out_alignment = as_vector(cg.incremental_forward());
  }

  return normalized_alignment_vector;
}

Expression AttentionalModel::GetContext(Expression& normalized_alignment_vector, const vector<Expression>& annotations, ComputationGraph& cg) {
  Expression annotation_matrix = concatenate_cols(annotations);
  return annotation_matrix * normalized_alignment_vector;
}

Expression AttentionalModel::GetNextOutputState(const Expression& context, const Expression& prev_target_word_embedding, ComputationGraph& cg) {
  return GetNextOutputState(output_builder.state(), context, prev_target_word_embedding, cg);
}

Expression AttentionalModel::GetNextOutputState(const RNNPointer& rnn_pointer,
    const Expression& context, const Expression& prev_target_word_embedding, ComputationGraph& cg) {
  // TODO: This could be optimized by allowing an RNNBuilder to take more than one input.
  // That would save us from having to store/use the off-(block-)diagonal elements in the
  // LSTM weight matrices.
  Expression state_rnn_input = concatenate({context, prev_target_word_embedding});
  Expression new_state = output_builder.add_input(rnn_pointer, state_rnn_input); // new_state = RNN(prev_state, prev_context, prev_target_word)
  return new_state;
}

Expression AttentionalModel::ComputeOutputDistribution(unsigned source_length, unsigned t, const Expression prev_target_embedding, const Expression state, const Expression context, const MLP& final_mlp, ComputationGraph& cg) {
  Expression output_distribution = final_mlp.Feed({prev_target_embedding, state, context});
  return output_distribution;
}

Expression AttentionalModel::ComputeNormalizedOutputDistribution(unsigned source_length, unsigned t, const Expression prev_target_embedding, const Expression state, const Expression context, const MLP& final_mlp, ComputationGraph& cg) {
  return log(softmax(ComputeOutputDistribution(source_length, t, prev_target_embedding, state, context, final_mlp, cg)));
}

Expression AttentionalModel::BuildGraphGivenAnnotations(const vector<Expression>& annotations, Expression zeroth_state, const vector<WordId>& target, ComputationGraph& cg) {
  // Target should always contain at least <s> and </s>
  assert (target.size() > 2);
  assert (target[0] == 1 && target[target.size() - 1] == 2);
  output_builder.new_graph(cg);
  output_builder.start_new_sequence();

  MLP aligner = GetAligner(cg);
  MLP final_mlp = GetFinalMLP(cg);
  const unsigned source_length = annotations.size();

  Expression prev_state = zeroth_state;
  vector<Expression> errors(target.size() - 1);
  // prev_word, t, prev_state, annotations, aligner, final_mlp, reference
  for (unsigned t = 1; t < target.size(); ++t) {
    WordId prev_word = target[t - 1];
    Expression prev_embedding = lookup(cg, p_Et, prev_word);
    Expression alignment = GetAlignments(t, prev_state, annotations, aligner, cg);
    Expression context = GetContext(alignment, annotations, cg);
    Expression state = GetNextOutputState(context, prev_embedding, cg);
    Expression distribution = ComputeOutputDistribution(source_length, t, prev_embedding, state, context, final_mlp, cg);
    errors[t - 1] = pickneglogsoftmax(distribution, target[t]);
    prev_state = state;
  }

  Expression total_error = sum(errors);
  return total_error;
}

MLP AttentionalModel::GetAligner(ComputationGraph& cg) const {
  Expression i_aIH1 = parameter(cg, p_aIH1);
  Expression i_aIH2 = parameter(cg, p_aIH2);
  Expression i_aHb = parameter(cg, p_aHb);
  Expression i_aHO = parameter(cg, p_aHO);
  Expression i_aOb = parameter(cg, p_aOb);
  MLP aligner = {{i_aIH1, i_aIH2}, i_aHb, i_aHO, i_aOb};
  return aligner;
}

MLP AttentionalModel::GetFinalMLP(ComputationGraph& cg) const {
  Expression i_fIH1 = parameter(cg, p_fIH1);
  Expression i_fIH2 = parameter(cg, p_fIH2);
  Expression i_fIH3 = parameter(cg, p_fIH3);
  Expression i_fHb = parameter(cg, p_fHb);
  Expression i_fHO = parameter(cg, p_fHO);
  Expression i_fOb = parameter(cg, p_fOb);
  MLP final_mlp = {{i_fIH1, i_fIH2, i_fIH3}, i_fHb, i_fHO, i_fOb};
  return final_mlp;
}

Expression AttentionalModel::GetZerothState(Expression zeroth_reverse_annotation, ComputationGraph& cg) const {
  Expression i_bs = parameter(cg, p_bs);
  Expression i_Ws = parameter(cg, p_Ws);
  Expression zeroth_state_untransformed = affine_transform({i_bs, i_Ws, zeroth_reverse_annotation});
  Expression zeroth_state = tanh(zeroth_state_untransformed);
  return zeroth_state;
}

Expression AttentionalModel::alignment_prior(unsigned t, unsigned source_length, ComputationGraph& cg) {
  vector<Expression> priors(source_length);
  Expression lambda = parameter(cg, p_tension);
  Expression len_mult = parameter(cg, p_length_multiplier);
  Expression target_len = source_length * len_mult;
  for (unsigned s = 0; s < source_length; ++s) {
    Expression num = input(cg, t + 1);
    Expression thing_inside_abs = cdiv(num, target_len) - (s + 1.0f) / source_length;
    Expression abs_val = max(thing_inside_abs, -thing_inside_abs);
    Expression p = -lambda * abs_val;
    priors[s] = p;
  }
  return softmax(concatenate(priors));
}

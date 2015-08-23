#include <queue>
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"

#include "bitext.h"
#include "attentional.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

AttentionalModel::AttentionalModel(Model& model, unsigned src_vocab_size, unsigned tgt_vocab_size) {
  forward_builder = LSTMBuilder(lstm_layer_count, embedding_dim, half_annotation_dim, &model);
  reverse_builder = LSTMBuilder(lstm_layer_count, embedding_dim, half_annotation_dim, &model);
  output_builder = LSTMBuilder(lstm_layer_count, embedding_dim + 2 * half_annotation_dim, output_state_dim, &model);
  tree_builder = TreeLSTMBuilder(5, lstm_layer_count, 2 * half_annotation_dim, 2 * half_annotation_dim, &model);
  p_Es = model.add_lookup_parameters(src_vocab_size, {embedding_dim});
  p_Et = model.add_lookup_parameters(tgt_vocab_size, {embedding_dim});
  p_aIH = model.add_parameters({alignment_hidden_dim, output_state_dim + 2 * half_annotation_dim});
  p_aHb = model.add_parameters({alignment_hidden_dim, 1});
  p_aHO = model.add_parameters({1, alignment_hidden_dim});
  p_aOb = model.add_parameters({1, 1});
  // The paper says s_0 = tanh(Ws * h1_reverse), and that Ws is an N x N matrix, but somehow below implies Ws is 2N x N.
  p_Ws = model.add_parameters({2 * half_annotation_dim, half_annotation_dim});
  p_bs = model.add_parameters({2 * half_annotation_dim});

  p_fIH = model.add_parameters({final_hidden_dim, embedding_dim + 2 * half_annotation_dim + output_state_dim});
  p_fHb = model.add_parameters({final_hidden_dim});
  p_fHO = model.add_parameters({tgt_vocab_size, final_hidden_dim});
  p_fOb = model.add_parameters({tgt_vocab_size});

  zero_annotation.resize(2 * half_annotation_dim);
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

OutputState AttentionalModel::GetNextOutputState(const Expression& prev_context, const Expression& prev_target_word_embedding,
    const vector<Expression>& annotations, const MLP& aligner, ComputationGraph& cg, vector<float>* out_alignment) {
  return GetNextOutputState(output_builder.state(), prev_context, prev_target_word_embedding, annotations, aligner, cg, out_alignment);
}

// TODO: If you ever want to optimize something, look no further. These concatenates can be removed by introducing new parameters
// or refactoring MLP into a proper class, leading to nice gains. Also there's gotta be a way to batch compute these alignment things
// rather than doing it one scalar at a time...
OutputState AttentionalModel::GetNextOutputState(const RNNPointer& rnn_pointer, const Expression& prev_context, const Expression& prev_target_word_embedding,
    const vector<Expression>& annotations, const MLP& aligner, ComputationGraph& cg, vector<float>* out_alignment) {
  const unsigned source_size = annotations.size();

  Expression state_rnn_input = concatenate({prev_context, prev_target_word_embedding});
  Expression new_state = output_builder.add_input(rnn_pointer, state_rnn_input); // new_state = RNN(prev_state, prev_context, prev_target_word)
  vector<Expression> unnormalized_alignments(source_size); // e_ij

  for (unsigned s = 0; s < source_size; ++s) {
    double prior = 1.0;
    Expression a_input = concatenate({new_state, annotations[s]});
    Expression a_hidden1 = affine_transform({aligner.i_Hb, aligner.i_IH, a_input});
    Expression a_hidden2 = tanh(a_hidden1);
    Expression a_output = affine_transform({aligner.i_Ob, aligner.i_HO, a_hidden2});
    unnormalized_alignments[s] = a_output * prior;
  }

  Expression unnormalized_alignment_vector = concatenate(unnormalized_alignments);
  Expression normalized_alignment_vector = softmax(unnormalized_alignment_vector); // \alpha_ij
  if (out_alignment != NULL) {
    *out_alignment = as_vector(cg.forward());
  }
  Expression annotation_matrix = concatenate_cols(annotations); // \alpha
  Expression context = annotation_matrix * normalized_alignment_vector; // c = \alpha * h

  OutputState os;
  os.state = new_state;
  os.context = context;
  os.rnn_pointer = output_builder.state();
  return os;
}

Expression AttentionalModel::ComputeOutputDistribution(const WordId prev_word, const Expression state, const Expression context, const MLP& final, ComputationGraph& cg) {
  Expression prev_target_embedding = lookup(cg, p_Et, prev_word);
  Expression final_input = concatenate({prev_target_embedding, state, context});
  Expression final_hidden1 = affine_transform({final.i_Hb, final.i_IH, final_input}); 
  Expression final_hidden2 = tanh({final_hidden1});
  Expression final_output = affine_transform({final.i_Ob, final.i_HO, final_hidden2});
  return final_output;
}

MLP AttentionalModel::GetAligner(ComputationGraph& cg) const {
  Expression i_aIH = parameter(cg, p_aIH);
  Expression i_aHb = parameter(cg, p_aHb);
  Expression i_aHO = parameter(cg, p_aHO);
  Expression i_aOb = parameter(cg, p_aOb);
  MLP aligner = {i_aIH, i_aHb, i_aHO, i_aOb};
  return aligner;
}

MLP AttentionalModel::GetFinalMLP(ComputationGraph& cg) const {
  Expression i_fIH = parameter(cg, p_fIH);
  Expression i_fHb = parameter(cg, p_fHb);
  Expression i_fHO = parameter(cg, p_fHO);
  Expression i_fOb = parameter(cg, p_fOb);
  MLP final_mlp = {i_fIH, i_fHb, i_fHO, i_fOb};
  return final_mlp;
}

Expression AttentionalModel::GetZerothContext(Expression zeroth_reverse_annotation, ComputationGraph& cg) const {
  Expression i_bs = parameter(cg, p_bs);
  Expression i_Ws = parameter(cg, p_Ws);
  Expression zeroth_context_untransformed = affine_transform({i_bs, i_Ws, zeroth_reverse_annotation});
  Expression zeroth_context = tanh(zeroth_context_untransformed);
  return zeroth_context;
}

OutputState AttentionalModel::GetInitialOutputState(Expression zeroth_context, const vector<Expression>& annotations, const MLP& aligner, const WordId kSOS, ComputationGraph& cg, vector<float>* alignment) {
  output_builder.start_new_sequence();
  Expression previous_target_word_embedding = lookup(cg, p_Et, kSOS);
  OutputState os = GetNextOutputState(zeroth_context, previous_target_word_embedding, annotations, aligner, cg, alignment);
  return os;
}

tuple<vector<Expression>, Expression> AttentionalModel::BuildAnnotationVectors(const vector<WordId>& source, ComputationGraph& cg) {
  vector<Expression> forward_annotations = BuildForwardAnnotations(source, cg);
  vector<Expression> reverse_annotations = BuildReverseAnnotations(source, cg);
  vector<Expression> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, cg);
  Expression zeroth_context = GetZerothContext(reverse_annotations[0], cg);
  return make_tuple(annotations, zeroth_context);
}

tuple<vector<Expression>, Expression> AttentionalModel::BuildAnnotationVectors(const SyntaxTree& source_tree, ComputationGraph& cg) {
  vector<Expression> linear_annotations;
  Expression zeroth_context;
  vector<WordId> source = source_tree.GetTerminals();
  tie(linear_annotations, zeroth_context) = BuildAnnotationVectors(source, cg);
  vector<Expression> annotations = BuildTreeAnnotationVectors(source_tree, linear_annotations, cg); 
  return make_tuple(annotations, zeroth_context);
}

Expression AttentionalModel::BuildGraphGivenAnnotations(const vector<Expression>& annotations, Expression zeroth_context, const vector<WordId>& target, ComputationGraph& cg) {
  // Target should always contain at least <s> and </s>
  assert (target.size() > 2);
  output_builder.new_graph(cg);
  output_builder.start_new_sequence();

  MLP aligner = GetAligner(cg);
  MLP final = GetFinalMLP(cg);

  vector<Expression> output_states(target.size());
  vector<Expression> contexts(target.size());
  contexts[0] = zeroth_context;

  for (unsigned t = 1; t < target.size(); ++t) {
    Expression prev_target_word_embedding = lookup(cg, p_Et, target[t - 1]);
    OutputState os = GetNextOutputState(contexts[t - 1], prev_target_word_embedding, annotations, aligner, cg);
    output_states[t] = os.state;
    contexts[t] = os.context;
  }

  vector<Expression> output_distributions(target.size() - 1);
  for (unsigned t = 1; t < target.size(); ++t) {
    WordId prev_word = target[t - 1];
    output_distributions[t - 1] = ComputeOutputDistribution(prev_word, output_states[t], contexts[t], final, cg);
  }

  vector<Expression> errors(target.size() - 1);
  for (unsigned t = 1; t < target.size(); ++t) {
    Expression output_distribution = output_distributions[t - 1];
    Expression error = pickneglogsoftmax(output_distribution, target[t]);
    errors[t - 1] = error;
  }
  Expression total_error = sum(errors);
  return total_error;
}

Expression AttentionalModel::BuildGraph(const vector<WordId>& source, const vector<WordId>& target, ComputationGraph& cg) {
  vector<Expression> annotations;
  Expression zeroth_context;
  tie(annotations, zeroth_context) = BuildAnnotationVectors(source, cg);

  return BuildGraphGivenAnnotations(annotations, zeroth_context, target, cg);
}

Expression AttentionalModel::BuildGraph(const SyntaxTree& source_tree, const vector<WordId>& target, ComputationGraph& cg) {
  vector<Expression> annotations;
  Expression zeroth_context;
  tie(annotations, zeroth_context) = BuildAnnotationVectors(source_tree, cg);

  return BuildGraphGivenAnnotations(annotations, zeroth_context, target, cg);
}

vector<Expression> AttentionalModel::BuildTreeAnnotationVectors(const SyntaxTree& source_tree, const vector<Expression>& linear_annotations, ComputationGraph& cg) {
  tree_builder.new_graph(cg);
  tree_builder.start_new_sequence();
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
        int child_id = (int)node->GetChild(j).id();
        assert (child_id < tree_annotations.size());
        children[j] = child_id;
      }

      Expression input_expr;
      if (node->NumChildren() == 0) {
        assert (terminal_index < linear_annotations.size());
        input_expr = linear_annotations[terminal_index];
        terminal_index++;
      }
      else {
        input_expr = input(cg, {(long)zero_annotation.size()}, &zero_annotation);
      }
      Expression node_annotation = tree_builder.add_input(children, input_expr);
      tree_annotations.push_back(node_annotation);
      //tree_annotations.push_back(lookup(cg, p_Es, node->label()));
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

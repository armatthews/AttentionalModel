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
  const unsigned source_size = annotations.size();

  Expression state_rnn_input = concatenate({prev_context, prev_target_word_embedding});
  Expression new_state = output_builder.add_input(state_rnn_input); // new_state = RNN(prev_state, prev_context, prev_target_word)
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

vector<vector<float> > AttentionalModel::Align(const vector<WordId>& source, const vector<WordId>& target) {
  ComputationGraph cg;
  output_builder.new_graph(cg);
  output_builder.start_new_sequence();

  vector<Expression> forward_annotations = BuildForwardAnnotations(source, cg);
  vector<Expression> reverse_annotations = BuildReverseAnnotations(source, cg);
  vector<Expression> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, cg);

  Expression i_aIH = parameter(cg, p_aIH);
  Expression i_aHb = parameter(cg, p_aHb);
  Expression i_aHO = parameter(cg, p_aHO);
  Expression i_aOb = parameter(cg, p_aOb);
  MLP aligner = {i_aIH, i_aHb, i_aHO, i_aOb};

  Expression i_fIH = parameter(cg, p_fIH);
  Expression i_fHb = parameter(cg, p_fHb);
  Expression i_fHO = parameter(cg, p_fHO);
  Expression i_fOb = parameter(cg, p_fOb);
  MLP final = {i_fIH, i_fHb, i_fHO, i_fOb};

  Expression i_bs = parameter(cg, p_bs);
  Expression i_Ws = parameter(cg, p_Ws);

  vector<Expression> output_states(target.size());
  vector<Expression> contexts(target.size());

  Expression zeroth_context_untransformed = affine_transform({i_bs, i_Ws, reverse_annotations[0]});
  Expression zeroth_context = tanh(zeroth_context_untransformed);
  Expression prev_context = zeroth_context;

  vector<vector<float> > alignment;
  for (unsigned t = 1; t < target.size() + 1; ++t) {
    vector<float> a;
    Expression prev_target_word_embedding = lookup(cg, p_Et, target[t - 1]);
    OutputState os = GetNextOutputState(prev_context, prev_target_word_embedding, annotations, aligner, cg, &a);
    prev_context = os.context;
    alignment.push_back(a);
  }
  return alignment;
}

vector<WordId> AttentionalModel::Translate(const vector<WordId>& source, WordId kSOS, WordId kEOS, unsigned beam_size, unsigned max_length) {
  KBestList<vector<WordId> > kbest = TranslateKBest(source, kSOS, kEOS, 1, beam_size, max_length);
  return kbest.hypothesis_list().begin()->second;
}

KBestList<vector<WordId> > AttentionalModel::TranslateKBest(const vector<WordId>& source, WordId kSOS, WordId kEOS, unsigned k, unsigned beam_size, unsigned max_length) {
  ComputationGraph cg;
  output_builder.new_graph(cg);

  vector<Expression> forward_annotations = BuildForwardAnnotations(source, cg);
  vector<Expression> reverse_annotations = BuildReverseAnnotations(source, cg);
  vector<Expression> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, cg);

  Expression i_aIH = parameter(cg, p_aIH);
  Expression i_aHb = parameter(cg, p_aHb);
  Expression i_aHO = parameter(cg, p_aHO);
  Expression i_aOb = parameter(cg, p_aOb);
  MLP aligner = {i_aIH, i_aHb, i_aHO, i_aOb};

  Expression i_fIH = parameter(cg, p_fIH);
  Expression i_fHb = parameter(cg, p_fHb);
  Expression i_fHO = parameter(cg, p_fHO);
  Expression i_fOb = parameter(cg, p_fOb);
  MLP final = {i_fIH, i_fHb, i_fHO, i_fOb};

  Expression i_bs = parameter(cg, p_bs);
  Expression i_Ws = parameter(cg, p_Ws);

  Expression zeroth_context_untransformed = affine_transform({i_bs, i_Ws, reverse_annotations[0]});
  Expression zeroth_context = tanh(zeroth_context_untransformed);
  /* Up until here this is all boiler plate */

  KBestList<vector<WordId> > completed_hyps(beam_size);
  KBestList<vector<WordId> > top_hyps(beam_size);
  top_hyps.add(0.0, {});

  // Invariant: each element in top_hyps should have a length of "length"
  for (unsigned length = 0; length < max_length; ++length) {
    KBestList<vector<WordId> > new_hyps(beam_size);
    for (auto scored_hyp : top_hyps.hypothesis_list()) {
      double score = scored_hyp.first;
      vector<WordId>& hyp = scored_hyp.second;
      assert (hyp.size() == length);

      // XXX: Rebuild the whole output state sequence
      output_builder.start_new_sequence(); 
      Expression previous_target_word_embedding = lookup(cg, p_Et, kSOS);
      OutputState os = GetNextOutputState(zeroth_context, previous_target_word_embedding, annotations, aligner, cg);
      for (WordId word : hyp) {
        assert (word != kEOS);
        Expression previous_target_word_embedding = lookup(cg, p_Et, word);
        os = GetNextOutputState(os.context, previous_target_word_embedding, annotations, aligner, cg);
      } 

      // Compute, normalize, and log the output distribution
      WordId prev_word = (hyp.size() > 0) ? hyp[hyp.size() - 1] : kSOS;
      Expression unnormalized_output_distribution = ComputeOutputDistribution(prev_word, os.state, os.context, final, cg);
      Expression output_distribution = softmax(unnormalized_output_distribution);
      Expression log_output_distribution = log(output_distribution);
      cerr << "HG has " << cg.nodes.size() << " nodes" << endl;
      vector<float> dist = as_vector(cg.incremental_forward());

      // Take the K best-looking words
      KBestList<WordId> best_words(beam_size);
      for (unsigned i = 0; i < dist.size(); ++i) {
        best_words.add(dist[i], i);
      }

      // For each of those K words, add it to the current hypothesis, and add the
      // resulting hyp to our kbest list, unless the new word is </s>,
      // in which case we add the new hyp to the list of completed hyps.
      for (pair<double, WordId> p : best_words.hypothesis_list()) {
        double word_score = p.first;
        WordId word = p.second;
        OutputState new_state = GetNextOutputState(os.context, previous_target_word_embedding, annotations, aligner, cg);
        output_builder.rewind_one_step();

        double new_score = score + word_score;
        vector<WordId> new_hyp = hyp;
        new_hyp.push_back(word);
        if (new_hyp.size() == max_length || word == kEOS) {
          completed_hyps.add(new_score, new_hyp);
        }
        else {
          new_hyps.add(new_score, new_hyp);
        }
      }
    }
    top_hyps = new_hyps;
  }
  return completed_hyps;
}

vector<WordId> AttentionalModel::SampleTranslation(const vector<WordId>& source, WordId kSOS, WordId kEOS, unsigned max_length) {
  ComputationGraph cg;
  output_builder.new_graph(cg);
  output_builder.start_new_sequence();

  vector<Expression> forward_annotations = BuildForwardAnnotations(source, cg);
  vector<Expression> reverse_annotations = BuildReverseAnnotations(source, cg);
  vector<Expression> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, cg);

  Expression i_aIH = parameter(cg, p_aIH);
  Expression i_aHb = parameter(cg, p_aHb);
  Expression i_aHO = parameter(cg, p_aHO);
  Expression i_aOb = parameter(cg, p_aOb);
  MLP aligner = {i_aIH, i_aHb, i_aHO, i_aOb};

  Expression i_fIH = parameter(cg, p_fIH);
  Expression i_fHb = parameter(cg, p_fHb);
  Expression i_fHO = parameter(cg, p_fHO);
  Expression i_fOb = parameter(cg, p_fOb);
  MLP final = {i_fIH, i_fHb, i_fHO, i_fOb};

  Expression i_bs = parameter(cg, p_bs);
  Expression i_Ws = parameter(cg, p_Ws);

  Expression zeroth_context_untransformed = affine_transform({i_bs, i_Ws, reverse_annotations[0]});
  Expression zeroth_context = tanh(zeroth_context_untransformed);

  Expression prev_context = zeroth_context;
  vector<WordId> output;
  unsigned prev_word = kSOS;
  while (prev_word != kEOS && output.size() < max_length) {
    Expression prev_target_word_embedding = lookup(cg, p_Et, prev_word);
    OutputState os = GetNextOutputState(prev_context, prev_target_word_embedding, annotations, aligner, cg);
    Expression log_output_distribution = ComputeOutputDistribution(prev_word, os.state, os.context, final, cg);
    Expression output_distribution = softmax(log_output_distribution);
    vector<float> dist = as_vector(cg.incremental_forward());
    double r = rand01();
    unsigned w = 0;
    while (true) {
      r -= dist[w];
      if (r < 0.0) {
        break;
      }
      ++w;
    }
    output.push_back(w);
    prev_word = w;
    prev_context = os.context;
  }
  return output;
}

Expression AttentionalModel::BuildGraph(const vector<WordId>& source, const vector<WordId>& target, ComputationGraph& cg) {
  // Target should always contain at least <s> and </s>
  assert (target.size() > 2);
  output_builder.new_graph(cg);
  output_builder.start_new_sequence();

  vector<Expression> forward_annotations = BuildForwardAnnotations(source, cg);
  vector<Expression> reverse_annotations = BuildReverseAnnotations(source, cg);
  vector<Expression> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, cg);

  Expression i_aIH = parameter(cg, p_aIH);
  Expression i_aHb = parameter(cg, p_aHb);
  Expression i_aHO = parameter(cg, p_aHO);
  Expression i_aOb = parameter(cg, p_aOb);
  MLP aligner = {i_aIH, i_aHb, i_aHO, i_aOb};

  Expression i_fIH = parameter(cg, p_fIH);
  Expression i_fHb = parameter(cg, p_fHb);
  Expression i_fHO = parameter(cg, p_fHO);
  Expression i_fOb = parameter(cg, p_fOb);
  MLP final = {i_fIH, i_fHb, i_fHO, i_fOb};

  Expression i_bs = parameter(cg, p_bs);
  Expression i_Ws = parameter(cg, p_Ws);

  vector<Expression> output_states(target.size());
  vector<Expression> contexts(target.size());

  // TODO: Verify that this crap corresponds to the comment below and is sane
  Expression zeroth_context_untransformed = affine_transform({i_bs, i_Ws, reverse_annotations[0]});
  contexts[0] = tanh(zeroth_context_untransformed);

  for (unsigned t = 1; t < target.size(); ++t) {
    Expression prev_target_word_embedding = lookup(cg, p_Et, target[t - 1]);
    OutputState os = GetNextOutputState(contexts[t - 1], prev_target_word_embedding, annotations, aligner, cg);
    output_states[t] = os.state;
    contexts[t] = os.context;
  }

  /*
  first input: W_s * h1_backwards
  afterwards, input: c_i
  c_i = sum_j(alpha_ij * h_j)
  alpha_ij = exp(e_ij) / sum_k(exp(e_ik))
  e_ij = a(s_i-1, h_j) where a is a FFNN
  */
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

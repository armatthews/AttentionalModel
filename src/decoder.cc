#include <iostream>
#include "decoder.h"
#include "utils.h"

AttentionalDecoder::AttentionalDecoder(AttentionalModel* model) {
  models.push_back(model);
}

AttentionalDecoder::AttentionalDecoder(const vector<AttentionalModel*>& models) : models(models) {
  assert (models.size() > 0);
}

void AttentionalDecoder::SetParams(unsigned max_length, WordId kSOS, WordId kEOS) {
  this->max_length = max_length;
  this->kSOS = kSOS;
  this->kEOS = kEOS;
}

vector<WordId> AttentionalDecoder::Translate(const vector<WordId>& source, unsigned beam_size) {
  KBestList<vector<WordId>> kbest = TranslateKBest(source, 1, beam_size);
  return kbest.hypothesis_list().begin()->second;
}

KBestList<vector<WordId>> AttentionalDecoder::TranslateKBest(const SyntaxTree& source_tree, unsigned K, unsigned beam_size) {
  ComputationGraph cg;

  vector<vector<Expression>> model_annotations;
  vector<MLP> model_aligners;
  vector<MLP> model_final_mlps;
  vector<OutputState> model_initial_states;

  for (AttentionalModel* model : models) {
    model->output_builder.new_graph(cg);
    vector<Expression> forward_annotations = model->BuildForwardAnnotations(source_tree.GetTerminals(), cg);
    vector<Expression> reverse_annotations = model->BuildReverseAnnotations(source_tree.GetTerminals(), cg);
    vector<Expression> linear_annotations = model->BuildAnnotationVectors(forward_annotations, reverse_annotations, cg);
    vector<Expression> annotations = model->BuildTreeAnnotationVectors(source_tree, linear_annotations, cg);
    Expression zeroth_context = model->GetZerothContext(reverse_annotations[0], cg);

    MLP aligner = model->GetAligner(cg);
    MLP final_mlp = model->GetFinalMLP(cg);

    OutputState os0 = model->GetInitialOutputState(zeroth_context, annotations, aligner, kSOS, cg);

    model_annotations.push_back(annotations);
    model_aligners.push_back(aligner);
    model_final_mlps.push_back(final_mlp);
    model_initial_states.push_back(os0);
  }

  KBestList<vector<WordId> > completed_hyps(K);
  KBestList<vector<PartialHypothesis>> top_hyps(beam_size);

  // XXX: We're storing the same word sequence N times
  vector<PartialHypothesis> initial_partial_hyps(models.size());
  for (unsigned i = 0; i < models.size(); ++i) {
    initial_partial_hyps[i] = {{}, model_initial_states[i]};
  }
  top_hyps.add(0.0, initial_partial_hyps);

  // Invariant: each element in top_hyps should have a length of "length"
  for (unsigned length = 0; length < max_length; ++length) {
    KBestList<vector<PartialHypothesis>> new_hyps(beam_size);
    for (auto scored_hyp : top_hyps.hypothesis_list()) {
      double score = scored_hyp.first;
      vector<PartialHypothesis>& hyp = scored_hyp.second;
      assert (hyp[0].words.size() == length);

      vector<Expression> model_log_distributions(models.size());
      for (unsigned i = 0; i < models.size(); ++i) {
        OutputState& os = hyp[i].state;

        // Compute, normalize, and log the output distribution
        WordId prev_word = (length > 0) ? hyp[i].words[length - 1] : kSOS;
        Expression unnormalized_output_distribution = models[i]->ComputeOutputDistribution(prev_word, os.state, os.context, model_final_mlps[i], cg);
        Expression output_distribution = softmax(unnormalized_output_distribution);
        Expression log_output_distribution = log(output_distribution);
        model_log_distributions[i] = log_output_distribution;
      }

      Expression overall_distribution = model_log_distributions[0];

      if (models.size() > 1) {
        overall_distribution = sum(model_log_distributions) / models.size();
        overall_distribution = log(softmax(overall_distribution)); // Renormalize
      }
      vector<float> dist = as_vector(cg.incremental_forward());

      // Take the K best-looking words
      KBestList<WordId> best_words(beam_size);
      for (unsigned j = 0; j < dist.size(); ++j) {
        best_words.add(dist[j], j);
      }

      // For each of those K words, add it to the current hypothesis, and add the
      // resulting hyp to our kbest list, unless the new word is </s>,
      // in which case we add the new hyp to the list of completed hyps.
      for (pair<double, WordId> p : best_words.hypothesis_list()) {
        double word_score = p.first;
        WordId word = p.second;
        double new_score = score + word_score;
        vector<PartialHypothesis> new_model_hyps(models.size());
        for (unsigned i = 0; i < models.size(); ++i) {
          OutputState& os = hyp[i].state;
          Expression previous_target_word_embedding = lookup(cg, models[i]->p_Et, word);
          OutputState new_state = models[i]->GetNextOutputState(os.rnn_pointer, os.context, previous_target_word_embedding, model_annotations[i], model_aligners[i], cg);
          PartialHypothesis new_hyp = {hyp[i].words, new_state};
          new_hyp.words.push_back(word);
          new_model_hyps[i] = new_hyp;
        }

        if (length + 1 == max_length || word == kEOS) {
          completed_hyps.add(new_score, new_model_hyps[0].words);
        }
        else {
          new_hyps.add(new_score, new_model_hyps);
        }
      }
    }
    top_hyps = new_hyps;
  }
  return completed_hyps;
}

KBestList<vector<WordId>> AttentionalDecoder::TranslateKBest(const vector<WordId>& source, unsigned K, unsigned beam_size) {
  ComputationGraph cg;

  vector<vector<Expression>> model_annotations;
  vector<MLP> model_aligners;
  vector<MLP> model_final_mlps;
  vector<OutputState> model_initial_states;

  for (AttentionalModel* model : models) {
    model->output_builder.new_graph(cg);
    vector<Expression> forward_annotations = model->BuildForwardAnnotations(source, cg);
    vector<Expression> reverse_annotations = model->BuildReverseAnnotations(source, cg);
    vector<Expression> annotations = model->BuildAnnotationVectors(forward_annotations, reverse_annotations, cg);
    Expression zeroth_context = model->GetZerothContext(reverse_annotations[0], cg);

    MLP aligner = model->GetAligner(cg);
    MLP final_mlp = model->GetFinalMLP(cg);

    OutputState os0 = model->GetInitialOutputState(zeroth_context, annotations, aligner, kSOS, cg);

    model_annotations.push_back(annotations);
    model_aligners.push_back(aligner);
    model_final_mlps.push_back(final_mlp);
    model_initial_states.push_back(os0);
  }

  KBestList<vector<WordId> > completed_hyps(beam_size);
  KBestList<vector<PartialHypothesis>> top_hyps(beam_size);

  // XXX: We're storing the same word sequence N times
  vector<PartialHypothesis> initial_partial_hyps(models.size());
  for (unsigned i = 0; i < models.size(); ++i) {
    initial_partial_hyps[i] = {{}, model_initial_states[i]};
  }
  top_hyps.add(0.0, initial_partial_hyps);

  // Invariant: each element in top_hyps should have a length of "length"
  for (unsigned length = 0; length < max_length; ++length) {
    KBestList<vector<PartialHypothesis>> new_hyps(beam_size);
    for (auto scored_hyp : top_hyps.hypothesis_list()) {
      double score = scored_hyp.first;
      vector<PartialHypothesis>& hyp = scored_hyp.second;
      assert (hyp[0].words.size() == length);

      vector<Expression> model_log_distributions(models.size());
      for (unsigned i = 0; i < models.size(); ++i) {
        OutputState& os = hyp[i].state;

        // Compute, normalize, and log the output distribution
        WordId prev_word = (length > 0) ? hyp[i].words[length - 1] : kSOS;
        Expression unnormalized_output_distribution = models[i]->ComputeOutputDistribution(prev_word, os.state, os.context, model_final_mlps[i], cg);
        Expression output_distribution = softmax(unnormalized_output_distribution);
        Expression log_output_distribution = log(output_distribution);
        model_log_distributions[i] = log_output_distribution;
      }

      Expression overall_distribution = model_log_distributions[0];

      if (models.size() > 1) {
        overall_distribution = sum(model_log_distributions) / models.size();
        overall_distribution = log(softmax(overall_distribution)); // Renormalize
      }
      vector<float> dist = as_vector(cg.incremental_forward());

      // Take the K best-looking words
      KBestList<WordId> best_words(beam_size);
      for (unsigned j = 0; j < dist.size(); ++j) {
        best_words.add(dist[j], j);
      }

      // For each of those K words, add it to the current hypothesis, and add the
      // resulting hyp to our kbest list, unless the new word is </s>,
      // in which case we add the new hyp to the list of completed hyps.
      for (pair<double, WordId> p : best_words.hypothesis_list()) {
        double word_score = p.first;
        WordId word = p.second;
        double new_score = score + word_score;
        vector<PartialHypothesis> new_model_hyps(models.size());
        for (unsigned i = 0; i < models.size(); ++i) {
          OutputState& os = hyp[i].state;
          Expression previous_target_word_embedding = lookup(cg, models[i]->p_Et, word);
          OutputState new_state = models[i]->GetNextOutputState(os.rnn_pointer, os.context, previous_target_word_embedding, model_annotations[i], model_aligners[i], cg);
          PartialHypothesis new_hyp = {hyp[i].words, new_state};
          new_hyp.words.push_back(word);
          new_model_hyps[i] = new_hyp;
        }

        if (length + 1 == max_length || word == kEOS) {
          completed_hyps.add(new_score, new_model_hyps[0].words);
        }
        else {
          new_hyps.add(new_score, new_model_hyps);
        }
      }
    }
    top_hyps = new_hyps;
  }
  return completed_hyps;
}

vector<vector<float>> AttentionalDecoder::Align(const vector<WordId>& source, const vector<WordId>& target) {
  ComputationGraph cg;
 
  vector<vector<Expression>> model_annotations;
  vector<MLP> model_aligners;
  vector<MLP> model_final_mlps;
  vector<OutputState> model_output_states;
  vector<vector<vector<float>>> model_alignments;

  for (AttentionalModel* model : models) {
    model->output_builder.new_graph(cg);
    model->output_builder.start_new_sequence();

    vector<Expression> forward_annotations = model->BuildForwardAnnotations(source, cg);
    vector<Expression> reverse_annotations = model->BuildReverseAnnotations(source, cg);
    vector<Expression> annotations = model->BuildAnnotationVectors(forward_annotations, reverse_annotations, cg);
    Expression zeroth_context = model->GetZerothContext(reverse_annotations[0], cg);

    MLP aligner = model->GetAligner(cg);
    MLP final_mlp = model->GetFinalMLP(cg);

    vector<float> a;
    OutputState os0 = model->GetInitialOutputState(zeroth_context, annotations, aligner, kSOS, cg, &a);

    model_annotations.push_back(annotations);
    model_aligners.push_back(aligner);
    model_final_mlps.push_back(final_mlp);
    model_output_states.push_back(os0);
    assert (a.size() == source.size());
    model_alignments.push_back({a});
  }
 
  for (unsigned t = 1; t < target.size() - 1; ++t) {
    for (unsigned i = 0; i < models.size(); ++i) {
      vector<float> a;
      Expression target_word_embedding = lookup(cg, models[i]->p_Et, target[t]);
      model_output_states[i] = models[i]->GetNextOutputState(model_output_states[i].context, target_word_embedding, model_annotations[i], model_aligners[i], cg, &a);
      assert (a.size() == source.size());
      model_alignments[i].push_back(a);
    }
  }
  vector<vector<float> > alignment(target.size());
  for (unsigned i = 0; i < target.size(); ++i) {
    alignment[i].resize(source.size());
  }

  for (unsigned i = 0; i < models.size(); ++i) {
    for (unsigned j = 0; j < target.size() - 1; ++j) {
      for (unsigned k = 0; k < source.size(); ++k) {
        alignment[j][k] += log(model_alignments[i][j][k]) / models.size();
      }
    }
  }

  for (unsigned j = 0; j < target.size() - 1; ++j) {
    float Z = logsumexp(alignment[j]);
    for (unsigned k = 0; k < source.size(); ++k) {
      alignment[j][k] = exp(alignment[j][k] - Z);
    }
  }

  return alignment;
}

/*
vector<WordId> AttentionalDecoder::SampleTranslation(const vector<WordId>& source) {
  ComputationGraph cg;
  output_builder.new_graph(cg);
  output_builder.start_new_sequence();

  vector<Expression> forward_annotations = BuildForwardAnnotations(source, cg);
  vector<Expression> reverse_annotations = BuildReverseAnnotations(source, cg);
  vector<Expression> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, cg);
  Expression zeroth_context = GetZerothContext(reverse_annotations[0], cg);

  MLP aligner = GetAligner(cg);
  MLP final = GetFinalMLP(cg);

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
}*/

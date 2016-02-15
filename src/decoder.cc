#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include "decoder.h"
#include "utils.h"

// Samples an item from a multinomial distribution
// The values in dist should sum to one.
unsigned Sample(const vector<float>& dist) {
  double r = rand01();
  unsigned w = 0;
  for (; w < dist.size(); ++w) {
    r -= dist[w];
    if (r < 0.0) {
      break;
    }
  }

  if (w == dist.size()) {
    --w;
  }
  return w;
}

DecoderState::DecoderState(unsigned n) {
  source_encodings.resize(n);
  output_states.resize(n);
}

Decoder::Decoder(Translator* translator) {
  translators.push_back(translator); 
}

Decoder::Decoder(const vector<Translator*>& translators) : translators(translators) {
  assert (translators.size() > 0);
}

void Decoder::SetParams(unsigned max_length, WordId kSOS, WordId kEOS) {
  this->max_length = max_length;
  this->kSOS = kSOS;
  this->kEOS = kEOS;
}

vector<Sentence> Decoder::SampleTranslations(const TranslatorInput* source, unsigned n) const {
  ComputationGraph cg;
  vector<Sentence> samples(n);
  vector<vector<Expression>> source_encodings(translators.size());

  for (unsigned i = 0; i < translators.size(); ++i) {
    translators[i]->NewGraph(cg);
    source_encodings[i] = translators[i]->Encode(source);
  }

  for (unsigned j = 0; j < n; ++j) {
    WordId prev_word = kSOS;
    Sentence sample;
    while (sample.size() < max_length) {
      vector<Expression> log_distributions(translators.size());
      for (unsigned i = 0; i < translators.size(); ++i) {
        Translator* translator = translators[i];
        vector<Expression>& encodings = source_encodings[i];
        Expression prev_state = translator->output_model->GetState();
        Expression context = translator->attention_model->GetContext(encodings, prev_state); 
        Expression new_state = translator->output_model->AddInput(prev_word, context);
        log_distributions[i] = translator->output_model->FullLogDistribution(new_state);
      }

      Expression final_distribution = softmax(sum(log_distributions));
      WordId w = Sample(final_distribution);
      sample.push_back(w);
      prev_word = w;
    }
    samples[j] = sample;
  }
  return samples;
}

vector<vector<WordId>> AttentionalDecoder::SampleTranslations(DecoderState& ds, unsigned n, ComputationGraph& cg) const {
  const unsigned source_length = ds.model_annotations[0].size();
  vector<vector<WordId>> outputs(n);
  for (unsigned j = 0; j < n; ++j) {
    for (unsigned i = 0; i < models.size(); ++i) {
      models[i]->output_builder.start_new_sequence();
    }
    vector<WordId> output;
    WordId prev_word = kSOS;
    while (prev_word != kEOS && output.size() < max_length) {
      unsigned t = output.size() + 1;
      vector<Expression> model_log_output_distributions(models.size());
      for (unsigned i = 0; i < models.size(); ++i) {
        AttentionalModel* model = models[i];
        vector<Expression>& annotations = ds.model_annotations[i];
        MLP& aligner = ds.model_aligners[i];
        MLP& final_mlp = ds.model_final_mlps[i];
        Expression prev_state = ds.model_output_states[i];

        Expression prev_embedding = lookup(cg, model->p_Et, prev_word);
        Expression alignment = model->GetAlignments(t, prev_state, annotations, aligner, cg);
        Expression context = model->GetContext(alignment, annotations, cg);
        Expression state = model->GetNextOutputState(context, prev_embedding, cg);
        model_log_output_distributions[i] = model->ComputeNormalizedOutputDistribution(source_length, t, prev_embedding, state, context, final_mlp, cg);
        ds.model_output_states[i] = state;
      }

      Expression total_log_output_distribution = sum(model_log_output_distributions);
      Expression output_distribution = softmax(total_log_output_distribution);
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
    }
    outputs[j] = output;
  }
  return outputs;
}

KBestList<vector<WordId>> AttentionalDecoder::TranslateKBest(DecoderState& ds, unsigned K, unsigned beam_size, ComputationGraph& cg) const {
  KBestList<vector<WordId> > completed_hyps(K);
  KBestList<vector<PartialHypothesis>> top_hyps(beam_size);
  const unsigned source_length = ds.model_annotations[0].size();

  // XXX: We're storing the same word sequence N times
  vector<PartialHypothesis> initial_partial_hyps(models.size());
  for (unsigned i = 0; i < models.size(); ++i) {
    initial_partial_hyps[i] = {{kSOS}, ds.model_output_states[i], models[i]->output_builder.state()};
  }
  top_hyps.add(0.0, initial_partial_hyps);

  // Invariant: each element in top_hyps should have a length of "t"
  for (unsigned t = 1; t <= max_length; ++t) {
    KBestList<vector<PartialHypothesis>> new_hyps(beam_size);
    for (auto scored_hyp : top_hyps.hypothesis_list()) {
      double score = scored_hyp.first;
      vector<PartialHypothesis>& hyp = scored_hyp.second;
      assert (hyp[0].words.size() == t);
      WordId prev_word = hyp[0].words.back();

      vector<Expression> model_log_output_distributions(models.size());
      for (unsigned i = 0; i < models.size(); ++i) {
        AttentionalModel* model = models[i];
        vector<Expression>& annotations = ds.model_annotations[i];
        MLP& aligner = ds.model_aligners[i];
        MLP& final_mlp = ds.model_final_mlps[i];
        Expression prev_state = hyp[i].state;

        Expression prev_embedding = lookup(cg, model->p_Et, prev_word);
        Expression alignment = model->GetAlignments(t, prev_state, annotations, aligner, cg);
        Expression context = model->GetContext(alignment, annotations, cg);
        Expression state = model->GetNextOutputState(hyp[i].rnn_pointer, context, prev_embedding, cg);
        model_log_output_distributions[i] = model->ComputeNormalizedOutputDistribution(source_length, t, prev_embedding, state, context, final_mlp, cg);
        ds.model_output_states[i] = state;
      }

      Expression overall_distribution = sum(model_log_output_distributions) / models.size();

      if (models.size() > 1) {
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
          PartialHypothesis new_hyp = {hyp[i].words, ds.model_output_states[i], models[i]->output_builder.state()};
          new_hyp.words.push_back(word);
          new_model_hyps[i] = new_hyp;
        }

        if (t + 1 == max_length || word == kEOS) {
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

vector<vector<cnn::real>> AttentionalDecoder::Align(DecoderState& ds, const vector<WordId>& target, ComputationGraph& cg) const {
  assert (target.size() >= 2 && target[0] == 1 && target[target.size() - 1] == 2);
  assert (ds.model_annotations.size() == models.size());
  assert (models.size() > 0);
  const unsigned source_size = ds.model_annotations[0].size();

  for (unsigned i = 0; i < models.size(); ++i) {
    AttentionalModel* model = models[i];
    vector<Expression>& annotations = ds.model_annotations[i];
    MLP& aligner = ds.model_aligners[i];
    Expression prev_state = ds.model_output_states[i];
    for (unsigned t = 1; t < target.size(); ++t) {
      vector<cnn::real> a;
      WordId prev_word = target[t - 1];
      Expression prev_embedding = lookup(cg, model->p_Et, prev_word);
      Expression alignment = model->GetAlignments(t, prev_state, annotations, aligner, cg, &a);
      Expression context = model->GetContext(alignment, annotations, cg);
      Expression state = model->GetNextOutputState(context, prev_embedding, cg);
      prev_state = state;

      assert (a.size() == source_size);
      ds.model_alignments[i].push_back(a);
    }
  }

  vector<vector<cnn::real>> alignment(target.size());
  for (unsigned i = 0; i < target.size(); ++i) {
    alignment[i].resize(source_size);
  }

  for (unsigned i = 0; i < models.size(); ++i) {
    for (unsigned j = 0; j < target.size() - 1; ++j) {
      for (unsigned k = 0; k < source_size; ++k) {
        alignment[j][k] += log(ds.model_alignments[i][j][k]) / models.size();
      }
    }
  }

  for (unsigned j = 0; j < target.size() - 1; ++j) {
    cnn::real Z = logsumexp(alignment[j]);
    for (unsigned k = 0; k < source_size; ++k) {
      alignment[j][k] = exp(alignment[j][k] - Z);
    }
  }

  return alignment;
}

vector<cnn::real> AttentionalDecoder::Loss(DecoderState& ds, const vector<WordId>& target, ComputationGraph& cg) const {
  assert (target.size() >= 2 && target[0] == 1 && target[target.size() - 1] == 2);
  vector<cnn::real> losses(target.size() - 1);
  const unsigned source_length = ds.model_annotations[0].size();

  for (unsigned i = 0; i < models.size(); ++i) {
    AttentionalModel* model = models[i];
    vector<Expression>& annotations = ds.model_annotations[i];
    MLP& aligner = ds.model_aligners[i];
    MLP& final_mlp = ds.model_final_mlps[i];
    Expression prev_state = ds.model_output_states[i];
    for (unsigned t = 1; t < target.size(); ++t) {
      WordId prev_word = target[t - 1];
      Expression prev_embedding = lookup(cg, model->p_Et, prev_word);
      Expression alignment = model->GetAlignments(t, prev_state, annotations, aligner, cg);
      Expression context = model->GetContext(alignment, annotations, cg);
      Expression state = model->GetNextOutputState(context, prev_embedding, cg);
      Expression distribution = model->ComputeOutputDistribution(source_length, t, prev_embedding, state, context, final_mlp, cg);
      Expression error = pickneglogsoftmax(distribution, target[t]);
      losses[t - 1] += as_scalar(cg.incremental_forward());
      prev_state = state;
    }
  }

  for (unsigned i = 0; i < losses.size(); ++i) {
    losses[i] /= models.size();
  }

  return losses;
}

tuple<vector<vector<Expression>>, vector<Expression>> AttentionalDecoder::InitializeAnnotations(const vector<WordId>& source, ComputationGraph& cg) const {
  vector<vector<Expression>> model_annotations;
  vector<Expression> model_zeroth_states;
  for (AttentionalModel* model : models) {
    vector<Expression> annotations;
    Expression zeroth_state;
    tie(annotations, zeroth_state) = model->BuildAnnotationVectors(source, cg);
    model_annotations.push_back(annotations);
    model_zeroth_states.push_back(zeroth_state);
  }
  return make_tuple(model_annotations, model_zeroth_states);
}

tuple<vector<vector<Expression>>, vector<Expression>> AttentionalDecoder::InitializeAnnotations(const SyntaxTree& source, ComputationGraph& cg) const {
  vector<vector<Expression>> model_annotations;
  vector<Expression> model_zeroth_states;
  for (AttentionalModel* model : models) {
    vector<Expression> annotations;
    Expression zeroth_state;
    tie(annotations, zeroth_state) = model->BuildAnnotationVectors(source, cg);
    model_annotations.push_back(annotations);
    model_zeroth_states.push_back(zeroth_state);
  }
  return make_tuple(model_annotations, model_zeroth_states);
}

DecoderState AttentionalDecoder::InitializeGivenAnnotations(const vector<vector<Expression>>& model_annotations, const vector<Expression> model_zeroth_states, ComputationGraph& cg) const {
  assert (model_annotations.size() == models.size());
  assert (model_zeroth_states.size() == models.size());

  vector<MLP> model_aligners;
  vector<MLP> model_final_mlps;
  vector<Expression> model_output_states;
  vector<vector<vector<cnn::real>>> model_alignments(models.size());

  for (unsigned i = 0; i < models.size(); ++i) {
    AttentionalModel* model = models[i];
    model->output_builder.new_graph(cg);
    model->output_builder.start_new_sequence();

    const Expression& zeroth_state = model_zeroth_states[i];

    MLP aligner = model->GetAligner(cg);
    MLP final_mlp = model->GetFinalMLP(cg);

    model_aligners.push_back(aligner);
    model_final_mlps.push_back(final_mlp);
    model_output_states.push_back(zeroth_state);
  }
  return { model_annotations, model_aligners, model_final_mlps, model_output_states, model_alignments };
}

DecoderState AttentionalDecoder::Initialize(const vector<WordId>& source, ComputationGraph& cg) const {
  assert (source.size() >= 2 && source[0] == 1 && source[source.size() - 1] == 2);
  vector<vector<Expression>> model_annotations;
  vector<Expression> model_zeroth_states;
  tie(model_annotations, model_zeroth_states) = InitializeAnnotations(source, cg);
  return InitializeGivenAnnotations(model_annotations, model_zeroth_states, cg);
}

DecoderState AttentionalDecoder::Initialize(const SyntaxTree& source, ComputationGraph& cg) const {
  vector<vector<Expression>> model_annotations;
  vector<Expression> model_zeroth_states;
  tie(model_annotations, model_zeroth_states) = InitializeAnnotations(source, cg);
  return InitializeGivenAnnotations(model_annotations, model_zeroth_states, cg);
}

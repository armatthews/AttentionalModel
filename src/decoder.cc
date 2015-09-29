#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
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

vector<vector<WordId>> AttentionalDecoder::SampleTranslations(const vector<WordId>& source, unsigned n) const {
  ComputationGraph cg;
  DecoderState ds = Initialize(source, cg);
  return SampleTranslations(ds, n, cg);
}

vector<vector<WordId>> AttentionalDecoder::SampleTranslations(const SyntaxTree& source_tree, unsigned n) const {
  ComputationGraph cg;
  DecoderState ds = Initialize(source_tree, cg);
  return SampleTranslations(ds, n, cg);
}

vector<WordId> AttentionalDecoder::Translate(const vector<WordId>& source, unsigned beam_size) const {
  KBestList<vector<WordId>> kbest = TranslateKBest(source, 1, beam_size);
  return kbest.hypothesis_list().begin()->second;
}

vector<WordId> AttentionalDecoder::Translate(const SyntaxTree& source, unsigned beam_size) const {
  KBestList<vector<WordId>> kbest = TranslateKBest(source, 1, beam_size);
  return kbest.hypothesis_list().begin()->second;
}

KBestList<vector<WordId>> AttentionalDecoder::TranslateKBest(const SyntaxTree& source_tree, unsigned K, unsigned beam_size) const {
  ComputationGraph cg;
  DecoderState ds = Initialize(source_tree, cg);
  return TranslateKBest(ds, K, beam_size, cg);
}

KBestList<vector<WordId>> AttentionalDecoder::TranslateKBest(const vector<WordId>& source, unsigned K, unsigned beam_size) const {
  ComputationGraph cg;
  DecoderState ds = Initialize(source, cg);
  return TranslateKBest(ds, K, beam_size, cg);
}

vector<vector<cnn::real>> AttentionalDecoder::Align(const vector<WordId>& source, const vector<WordId>& target) const {
  ComputationGraph cg;
  DecoderState ds = Initialize(source, cg);
  return Align(ds, target, cg);
}

vector<vector<cnn::real>> AttentionalDecoder::Align(const SyntaxTree& source, const vector<WordId>& target) const {
  ComputationGraph cg;
  DecoderState ds = Initialize(source, cg);
  return Align(ds, target, cg);
}

vector<cnn::real> AttentionalDecoder::Loss(const vector<WordId>& source, const vector<WordId>& target) const {
  ComputationGraph cg;
  DecoderState ds = Initialize(source, cg);
  return Loss(ds, target, cg);
}

vector<cnn::real> AttentionalDecoder::Loss(const SyntaxTree& source, const vector<WordId>& target) const {
  ComputationGraph cg;
  DecoderState ds = Initialize(source, cg);
  return Loss(ds, target, cg);
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

tuple<Dict, Dict, Model*, AttentionalModel*> LoadModel(const string& model_filename) {
  Dict source_vocab;
  Dict target_vocab;
  ifstream model_file(model_filename);
  if (!model_file.is_open()) {
    cerr << "ERROR: Unable to open " << model_filename << endl;
    exit(1);
  }
  boost::archive::text_iarchive ia(model_file);

  ia & source_vocab;
  ia & target_vocab;
  source_vocab.Freeze();
  target_vocab.Freeze();

  Model* cnn_model = new Model();
  //AttentionalModel* attentional_model = new AttentionalModel(*cnn_model, source_vocab.size(), target_vocab.size());
  AttentionalModel* attentional_model = new AttentionalModel();

  ia & *attentional_model;
  attentional_model->InitializeParameters(*cnn_model, source_vocab.size(), target_vocab.size());

  ia & *cnn_model;

  return make_tuple(source_vocab, target_vocab, cnn_model, attentional_model);
}

tuple<Dict, Dict, vector<Model*>, vector<AttentionalModel*>> LoadModels(const vector<string>& model_filenames) {
  vector<Model*> cnn_models;
  vector<AttentionalModel*> attentional_models;
  // XXX: We just use the last set of dictionaries, assuming they're all the same
  Dict source_vocab;
  Dict target_vocab;
  Model* cnn_model = nullptr;
  AttentionalModel* attentional_model = nullptr;
  for (const string& model_filename : model_filenames) {
    tie(source_vocab, target_vocab, cnn_model, attentional_model) = LoadModel(model_filename);
    cnn_models.push_back(cnn_model);
    attentional_models.push_back(attentional_model);
  }
  return make_tuple(source_vocab, target_vocab, cnn_models, attentional_models);
}

tuple<vector<WordId>, vector<WordId>> ReadInputLine(const string& line, Dict& source_vocab, Dict& target_vocab) {
  vector<string> parts = tokenize(line, "|||");
  parts = strip(parts);
  assert (parts.size() == 1 || parts.size() == 2);

  WordId ksSOS = source_vocab.Convert("<s>");
  WordId ksEOS = source_vocab.Convert("</s>");
  WordId ktSOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");

  vector<WordId> source = ReadSentence(parts[0], &source_vocab);
  source.insert(source.begin(), ksSOS);
  source.insert(source.end(), ksEOS);

  vector<WordId> target;
  if (parts.size() > 1) {
    target = ReadSentence(parts[1], &target_vocab);
    target.insert(target.begin(), ktSOS);
    target.push_back(ktEOS);
  }

  cerr << "Read input:";
  for (WordId w: source) {
    cerr << " " << source_vocab.Convert(w);
  }
  cerr << " |||";
  for (WordId w: target) {
    cerr << " " << target_vocab.Convert(w);
  }
  cerr << endl;

  return make_tuple(source, target);
}

tuple<SyntaxTree, vector<WordId>> ReadT2SInputLine(const string& line, Dict& source_vocab, Dict& target_vocab) {
  vector<string> parts = tokenize(line, "|||");
  parts = strip(parts);
  assert (parts.size() == 1 || parts.size() == 2);

  WordId ktSOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");

  SyntaxTree source_tree(parts[0], &source_vocab);
  source_tree.AssignNodeIds();

  vector<WordId> target;
  if (parts.size() > 1) {
    target = ReadSentence(parts[1], &target_vocab);
    target.insert(target.begin(), ktSOS);
    target.push_back(ktEOS);
  }

  cerr << "Read tree input:";
  for (WordId w: source_tree.GetTerminals()) {
    cerr << " " << source_vocab.Convert(w);
  }
  cerr << " |||";
  for (WordId w: target) {
    cerr << " " << target_vocab.Convert(w);
  }
  cerr << endl;

  return make_tuple(source_tree, target);
}

#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <iostream>
#include <fstream>
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

vector<WordId> AttentionalDecoder::SampleTranslation(const vector<WordId>& source) {
  ComputationGraph cg;
  DecoderState ds = Initialize(source, cg);
  return SampleTranslation(ds, cg);
}

vector<WordId> AttentionalDecoder::SampleTranslation(const SyntaxTree& source_tree) {
  ComputationGraph cg;
  DecoderState ds = Initialize(source_tree, cg);
  return SampleTranslation(ds, cg);
}

vector<WordId> AttentionalDecoder::Translate(const vector<WordId>& source, unsigned beam_size) {
  KBestList<vector<WordId>> kbest = TranslateKBest(source, 1, beam_size);
  return kbest.hypothesis_list().begin()->second;
}

vector<WordId> AttentionalDecoder::Translate(const SyntaxTree& source, unsigned beam_size) {
  KBestList<vector<WordId>> kbest = TranslateKBest(source, 1, beam_size);
  return kbest.hypothesis_list().begin()->second;
}

KBestList<vector<WordId>> AttentionalDecoder::TranslateKBest(const SyntaxTree& source_tree, unsigned K, unsigned beam_size) {
  ComputationGraph cg;
  DecoderState ds = Initialize(source_tree, cg);
  return TranslateKBest(ds, K, beam_size, cg);
}

KBestList<vector<WordId>> AttentionalDecoder::TranslateKBest(const vector<WordId>& source, unsigned K, unsigned beam_size) {
  ComputationGraph cg;
  DecoderState ds = Initialize(source, cg);
  return TranslateKBest(ds, K, beam_size, cg);
}

vector<vector<float>> AttentionalDecoder::Align(const vector<WordId>& source, const vector<WordId>& target) {
  ComputationGraph cg;
  DecoderState ds = Initialize(source, cg);
  return Align(ds, target, cg);
}

vector<vector<float>> AttentionalDecoder::Align(const SyntaxTree& source, const vector<WordId>& target) {
  ComputationGraph cg;
  DecoderState ds = Initialize(source, cg);
  return Align(ds, target, cg);
}

vector<WordId> AttentionalDecoder::SampleTranslation(DecoderState& ds, ComputationGraph& cg) {
  vector<WordId> output;
  unsigned prev_word = kSOS;
  while (prev_word != kEOS && output.size() < max_length) {
    vector<Expression> model_log_output_distributions(models.size());
    for (unsigned i = 0; i < models.size(); ++i) {
      OutputState& os = ds.model_output_states[i];
      Expression prev_target_word_embedding = lookup(cg, models[i]->p_Et, prev_word);
      os = models[i]->GetNextOutputState(os.context, prev_target_word_embedding, ds.model_annotations[i], ds.model_aligners[i], cg);
      model_log_output_distributions[i] = models[i]->ComputeOutputDistribution(prev_word, os.state, os.context, ds.model_final_mlps[i], cg);
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
  return output;
}

KBestList<vector<WordId>> AttentionalDecoder::TranslateKBest(DecoderState& ds, unsigned K, unsigned beam_size, ComputationGraph& cg) {
  KBestList<vector<WordId> > completed_hyps(beam_size);
  KBestList<vector<PartialHypothesis>> top_hyps(beam_size);

  // XXX: We're storing the same word sequence N times
  vector<PartialHypothesis> initial_partial_hyps(models.size());
  for (unsigned i = 0; i < models.size(); ++i) {
    initial_partial_hyps[i] = {{}, ds.model_output_states[i]};
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
        Expression unnormalized_output_distribution = models[i]->ComputeOutputDistribution(prev_word, os.state, os.context, ds.model_final_mlps[i], cg);
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
          OutputState new_state = models[i]->GetNextOutputState(os.rnn_pointer, os.context, previous_target_word_embedding, ds.model_annotations[i], ds.model_aligners[i], cg);
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

vector<vector<float>> AttentionalDecoder::Align(DecoderState& ds, const vector<WordId>& target, ComputationGraph& cg) {
  assert (ds.model_annotations.size() == models.size());
  assert (models.size() > 0);
  assert (ds.model_alignments[0].size() == 1);
  const unsigned source_size = ds.model_annotations[0].size();

  for (unsigned t = 1; t < target.size(); ++t) {
    for (unsigned i = 0; i < models.size(); ++i) {
      vector<float> a;
      Expression target_word_embedding = lookup(cg, models[i]->p_Et, target[t]);
      ds.model_output_states[i] = models[i]->GetNextOutputState(ds.model_output_states[i].context, target_word_embedding, ds.model_annotations[i], ds.model_aligners[i], cg, &a);
      assert (a.size() == source_size);
      ds.model_alignments[i].push_back(a);
    }
  }

  vector<vector<float> > alignment(target.size());
  for (unsigned i = 0; i < target.size(); ++i) {
    alignment[i].resize(source_size);
  }

  for (unsigned i = 0; i < models.size(); ++i) {
    for (unsigned j = 0; j < target.size(); ++j) {
      for (unsigned k = 0; k < source_size; ++k) {
        alignment[j][k] += log(ds.model_alignments[i][j][k]) / models.size();
      }
    }
  }

  for (unsigned j = 0; j < target.size(); ++j) {
    float Z = logsumexp(alignment[j]);
    for (unsigned k = 0; k < source_size; ++k) {
      alignment[j][k] = exp(alignment[j][k] - Z);
    }
  }

  return alignment;
}

vector<cnn::real> AttentionalDecoder::Loss(DecoderState& source, const vector<WordId>& target, ComputationGraph& cg) {
  vector<cnn::real> losses;
  return losses;
}

tuple<vector<vector<Expression>>, vector<Expression>> AttentionalDecoder::InitializeAnnotations(const vector<WordId>& source, ComputationGraph& cg) {
  vector<vector<Expression>> model_annotations;
  vector<Expression> model_zeroth_contexts;
  for (AttentionalModel* model : models) {
    vector<Expression> annotations;
    Expression zeroth_context;
    tie(annotations, zeroth_context) = model->BuildAnnotationVectors(source, cg);
    model_annotations.push_back(annotations);
    model_zeroth_contexts.push_back(zeroth_context);
  }
  return make_tuple(model_annotations, model_zeroth_contexts);
}

tuple<vector<vector<Expression>>, vector<Expression>> AttentionalDecoder::InitializeAnnotations(const SyntaxTree& source, ComputationGraph& cg) {
  vector<vector<Expression>> model_annotations;
  vector<Expression> model_zeroth_contexts;
  for (AttentionalModel* model : models) {
    vector<Expression> annotations;
    Expression zeroth_context;
    tie(annotations, zeroth_context) = model->BuildAnnotationVectors(source, cg);
    model_annotations.push_back(annotations);
    model_zeroth_contexts.push_back(zeroth_context);
  }
  return make_tuple(model_annotations, model_zeroth_contexts);
}

DecoderState AttentionalDecoder::InitializeGivenAnnotations(const vector<vector<Expression>>& model_annotations, const vector<Expression> model_zeroth_contexts, ComputationGraph& cg) { 
  assert (model_annotations.size() == models.size());
  assert (model_zeroth_contexts.size() == models.size());

  vector<MLP> model_aligners;
  vector<MLP> model_final_mlps;
  vector<OutputState> model_output_states;
  vector<vector<vector<float>>> model_alignments;

  for (unsigned i = 0; i < models.size(); ++i) {
    AttentionalModel* model = models[i];
    model->output_builder.new_graph(cg);
    model->output_builder.start_new_sequence();

    const vector<Expression>& annotations = model_annotations[i];
    const Expression& zeroth_context = model_zeroth_contexts[i];

    MLP aligner = model->GetAligner(cg);
    MLP final_mlp = model->GetFinalMLP(cg);

    vector<float> a;
    OutputState os0 = model->GetInitialOutputState(zeroth_context, annotations, aligner, kSOS, cg, &a);

    model_aligners.push_back(aligner);
    model_final_mlps.push_back(final_mlp);
    model_output_states.push_back(os0);
    assert (a.size() == annotations.size());
    model_alignments.push_back({a});
  }
  return { model_annotations, model_aligners, model_final_mlps, model_output_states, model_alignments };
}

DecoderState AttentionalDecoder::Initialize(const vector<WordId>& source, ComputationGraph& cg) { 
  vector<vector<Expression>> model_annotations;
  vector<Expression> model_zeroth_contexts;
  tie(model_annotations, model_zeroth_contexts) = InitializeAnnotations(source, cg);
  return InitializeGivenAnnotations(model_annotations, model_zeroth_contexts, cg);
}

DecoderState AttentionalDecoder::Initialize(const SyntaxTree& source, ComputationGraph& cg) { 
  vector<vector<Expression>> model_annotations;
  vector<Expression> model_zeroth_contexts;
  tie(model_annotations, model_zeroth_contexts) = InitializeAnnotations(source, cg);
  return InitializeGivenAnnotations(model_annotations, model_zeroth_contexts, cg);
}

tuple<Dict, Dict, vector<Model*>, vector<AttentionalModel*>> LoadModels(const vector<string>& model_filenames) {
  vector<Model*> cnn_models;
  vector<AttentionalModel*> attentional_models;
  // XXX: We just use the last set of dictionaries, assuming they're all the same
  Dict source_vocab;
  Dict target_vocab;
  for (const string& model_filename : model_filenames) {
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

    Model* model = new Model();
    AttentionalModel* attentional_model = new AttentionalModel(*model, source_vocab.size(), target_vocab.size());
    cnn_models.push_back(model);
    attentional_models.push_back(attentional_model);

    ia & *attentional_model;
    ia & *model;
  }
  return make_tuple(source_vocab, target_vocab, cnn_models, attentional_models);
}

tuple<vector<WordId>, vector<WordId>> ReadInputLine(const string& line, Dict& source_vocab, Dict& target_vocab) {
  vector<string> parts = tokenize(line, "|||");
  parts = strip(parts);
  assert (parts.size() == 1 || parts.size() == 2);

  WordId ksSOS = source_vocab.Convert("<s>");
  WordId ksEOS = source_vocab.Convert("</s>");
  WordId ktEOS = target_vocab.Convert("</s>");

  vector<WordId> source = ReadSentence(parts[0], &source_vocab);
  source.insert(source.begin(), ksSOS);
  source.insert(source.end(), ksEOS);

  vector<WordId> target;
  if (parts.size() > 1) {
    target = ReadSentence(parts[1], &target_vocab);
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

  WordId ktEOS = target_vocab.Convert("</s>");

  SyntaxTree source_tree(parts[0], &source_vocab);
  source_tree.AssignNodeIds();

  vector<WordId> target;
  if (parts.size() > 1) {
    target = ReadSentence(parts[1], &target_vocab);
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

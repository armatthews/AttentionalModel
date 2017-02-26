#include "translator.h"

Translator::Translator() {}

Translator::Translator(EncoderModel* encoder, AttentionModel* attention, OutputModel* output) {
  encoder_model = encoder;
  attention_model = attention;
  output_model = output;
}

void Translator::NewGraph(ComputationGraph& cg) {
  encoder_model->NewGraph(cg);
  attention_model->NewGraph(cg);
  output_model->NewGraph(cg);
}

void Translator::SetDropout(float rate) {
  encoder_model->SetDropout(rate);
  attention_model->SetDropout(rate);
  output_model->SetDropout(rate);
}

Expression Translator::BuildGraph(const InputSentence* const source, const OutputSentence* const target, ComputationGraph& cg) {
  NewGraph(cg);
  vector<Expression> word_losses(target->size());

  const InputSentence* const linear_source = dynamic_cast<const SentWithTree*>(source)->sent;
  const SyntaxTree* const source_tree = dynamic_cast<const SentWithTree*>(source)->tree;
  vector<Expression> encodings = encoder_model->Encode(linear_source);
  //cerr << "Input encodings have length " << encodings.size() << endl;

  TreeAttentionModel* att = dynamic_cast<TreeAttentionModel*>(attention_model); 

  Expression state = output_model->GetState();
  attention_model->NewSentence(source);
  for (unsigned i = 0; i < target->size(); ++i) {
    const shared_ptr<Word> word = target->at(i);
    assert (same_value(state, output_model->GetState()));
    word_losses[i] = output_model->Loss(state, word);

    Expression context = att->GetContext(encodings, source_tree, state);
    state = output_model->AddInput(word, context);
  }
  return sum(word_losses);
}

void Translator::Sample(const vector<Expression>& encodings, shared_ptr<OutputSentence> prefix, float prefix_score, RNNPointer state_pointer, unsigned sample_count, unsigned max_length, ComputationGraph& cg, vector<pair<shared_ptr<OutputSentence>, float>>& samples) {
  if (max_length == 0) {
    shared_ptr<OutputSentence> sample = make_shared<OutputSentence>(*prefix);
    samples.push_back(make_pair(sample, prefix_score));
    return;
  }

  Expression output_state = output_model->GetState(state_pointer);
  Expression context = attention_model->GetContext(encodings, output_state);

  unordered_map<shared_ptr<Word>, unsigned> continuations;
  unordered_map<shared_ptr<Word>, float> scores;
  for (unsigned i = 0; i < sample_count; ++i) {
    pair<shared_ptr<Word>, float> sample = output_model->Sample(state_pointer, output_state);
    shared_ptr<Word> w = get<0>(sample);
    float score = get<1>(sample);
    if (continuations.find(w) != continuations.end()) {
      continuations[w]++;
    }
    else {
      continuations[w] = 1;
      scores[w] = score;
    }
  }

  for (auto it = continuations.begin(); it != continuations.end(); ++it) {
    shared_ptr<Word> w = it->first;
    float score = prefix_score + scores[w];
    prefix->push_back(w);
    output_model->AddInput(w, context, state_pointer);
    RNNPointer new_pointer = output_model->GetStatePointer();

    if (output_model->IsDone()) {
      for (unsigned i = 0; i < it->second; ++i) {
        shared_ptr<OutputSentence> sample = make_shared<OutputSentence>(*prefix);
        samples.push_back(make_pair(sample, score));
      }
    }
    else {
      Sample(encodings, prefix, score, new_pointer, it->second, max_length - 1, cg, samples);
    }
    prefix->pop_back();
  }
}

vector<pair<shared_ptr<OutputSentence>, float>> Translator::Sample(const InputSentence* const source, unsigned sample_count, unsigned max_length) {
  ComputationGraph cg;
  NewGraph(cg);
  vector<Expression> encodings = encoder_model->Encode(source);
  attention_model->NewSentence(source);

  shared_ptr<OutputSentence> prefix = make_shared<OutputSentence>();
  vector<pair<shared_ptr<OutputSentence>, float>> samples;
  Sample(encodings, prefix, 0.0f, output_model->GetStatePointer(), sample_count, max_length, cg, samples);
  return samples;
}

vector<Expression> Translator::Align(const InputSentence* const source, const OutputSentence* const target, ComputationGraph& cg) {
  NewGraph(cg);
  vector<Expression> encodings = encoder_model->Encode(source);
  vector<Expression> alignments;
  Expression input_matrix = concatenate_cols(encodings);
  for (unsigned i = 1; i < target->size(); ++i) {
    const shared_ptr<Word> prev_word = (*target)[i - 1];
    Expression state = output_model->GetState();
    Expression word_alignment = attention_model->GetAlignmentVector(encodings, state);
    Expression context = input_matrix * word_alignment;
    output_model->AddInput(prev_word, context);
    alignments.push_back(word_alignment);
  }
  return alignments;
}

KBestList<shared_ptr<OutputSentence>> Translator::Translate(const InputSentence* const source, unsigned K, unsigned beam_size, unsigned max_length) {
  assert (beam_size >= K);
  ComputationGraph cg;
  NewGraph(cg);

  KBestList<shared_ptr<OutputSentence>> complete_hyps(K);
  KBestList<pair<shared_ptr<OutputSentence>, RNNPointer>> top_hyps(beam_size);
  top_hyps.add(0.0, make_pair(make_shared<OutputSentence>(), output_model->GetStatePointer()));

  vector<Expression> encodings = encoder_model->Encode(source);
  attention_model->NewSentence(source);

  for (unsigned length = 0; length < max_length; ++length) {
    KBestList<pair<shared_ptr<OutputSentence>, RNNPointer>> new_hyps(beam_size);

    for (auto& hyp : top_hyps.hypothesis_list()) {
      double hyp_score = get<0>(hyp);

      // Early termination: if we have K completed hypotheses, and the current prefix's
      // score is worse than the worst of the complete hyps, then it's impossible to
      // later get a better hypothesis later.
      // Assumption: the log prob of every word will be <= buffer.
      // This is true for 0.0 by default, but length priors and such may invalidate this assumption.
      const float buffer = 0.0;
      if (complete_hyps.size() >= K && hyp_score < complete_hyps.worst_score() - buffer) {
        break;
      }

      shared_ptr<OutputSentence> hyp_sentence = get<0>(get<1>(hyp));
      RNNPointer state_pointer = get<1>(get<1>(hyp));
      assert (hyp_sentence->size() == length);
      Expression output_state = output_model->GetState(state_pointer);
      Expression context = attention_model->GetContext(encodings, output_state);
      KBestList<shared_ptr<Word>> best_words = output_model->PredictKBest(state_pointer, output_state, beam_size);

      for (auto& w : best_words.hypothesis_list()) {
        double word_score = get<0>(w);
        shared_ptr<Word> word = get<1>(w);
        double new_score = hyp_score + word_score;
        shared_ptr<OutputSentence> new_sentence(new OutputSentence(*hyp_sentence));
        new_sentence->push_back(word);
        output_model->AddInput(word, context, state_pointer);
        if (!output_model->IsDone()) {
          new_hyps.add(new_score, make_pair(new_sentence, output_model->GetStatePointer()));
        }
        else {
          complete_hyps.add(new_score, new_sentence);
        }
      }
    }
    top_hyps = new_hyps;
  }

  for (auto& hyp : top_hyps.hypothesis_list()) {
    double score = get<0>(hyp);
    shared_ptr<OutputSentence> sentence = get<0>(get<1>(hyp));
    complete_hyps.add(score, sentence);
  }
  return complete_hyps;
}

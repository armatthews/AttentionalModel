#include "translator.h"
#include "length_priors.h"

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

  vector<Expression> encodings = encoder_model->Encode(source);

  Expression state = output_model->GetState();
  attention_model->NewSentence(source);
  for (unsigned i = 0; i < target->size(); ++i) {
    const shared_ptr<Word> word = target->at(i);
    assert (same_value(state, output_model->GetState()));
    word_losses[i] = output_model->Loss(state, word);

    Expression context = attention_model->GetContext(encodings, state);
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

    // TODO: We can bail early if the best thing in top_hyps is worse than the lowest ting
    // in complete_hyps, and complete_hyps.size() > K, since the scores are always negative. (assert this!)

    for (auto& hyp : top_hyps.hypothesis_list()) {
      double hyp_score = get<0>(hyp);
      const float c = -0.2;
      //const float buffer = 0.0f;
      //const float buffer = 11.1f * c;
      const float buffer = max(c, 0.0f);
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
        assert (word_score <= 0);
        shared_ptr<Word> word = get<1>(w);
        double new_score = hyp_score + word_score;
        shared_ptr<OutputSentence> new_sentence(new OutputSentence(*hyp_sentence));
        new_sentence->push_back(word);
        output_model->AddInput(word, context, state_pointer);

        // XXX: Length priors are incredibly broken for RNNGs
        assert (source->NumNodes() >= 2);
        unsigned real_source_length = source->NumNodes() - 2; // Remove <s> and </s>
        int real_target_length = (int)length - 1; // Remove <s>
        if (!output_model->IsDone()) {
          // Length prior type 1 (xxx): Use a (smoothed) multinomial table learned from the training corpus.
          // At each time step, subtract off the previous bonus/penalty and add in the new one
          /*assert (real_source_length < 60);
          assert (real_target_length < 66);
          new_score += c * length_priors[real_source_length][real_target_length];
          if (real_target_length > 0) {
            new_score -= c * length_priors[real_source_length][real_target_length - 1]; // XXX: length_priors is global and is made for BTEC
          }*/

          // Length prior type 2 (yyy): Just add a fixed bonus for each word generated
          // XXX: This could break things if the total score per word goes > 0.0, due to the early stopping thing
          new_score += c;
          new_hyps.add(new_score, make_pair(new_sentence, output_model->GetStatePointer()));
        }
        else {
          // Length prior type 3 (zzz): Use the smoothed multinomial, but only add it it once, at the end.
          /*assert (real_source_length < 60);
          assert (real_target_length < 66);
          new_score += c * length_priors[real_source_length][real_target_length];*/
          complete_hyps.add(new_score, new_sentence);
        }
      }
    }
    top_hyps = new_hyps;
  }

  // XXX: Uncomment this block to re-allow max-length hyps not terminated by </s>
  /*for (auto& hyp : top_hyps.hypothesis_list()) {
    double score = get<0>(hyp);
    shared_ptr<OutputSentence> sentence = get<0>(get<1>(hyp));
    complete_hyps.add(score, sentence);
  }*/
  return complete_hyps;
}

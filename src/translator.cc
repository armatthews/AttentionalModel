#include "translator.h"

Translator::Translator() {}

Translator::Translator(EncoderModel* encoder, AttentionModel* attention, OutputModel* output) {
  encoder_model = encoder;
  attention_model = attention;
  output_model = output;
}

bool Translator::IsT2S() const {
  return encoder_model->IsT2S();
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

Expression Translator::BuildGraph(const TranslatorInput* const source, const Sentence& target, ComputationGraph& cg) {
  NewGraph(cg);
  vector<Expression> word_losses(target.size());
  word_losses[0] = input(cg, 0.0f); // <s>

  vector<Expression> encodings = encoder_model->Encode(source);
  for (unsigned i = 1; i < target.size(); ++i) {
    const WordId& prev_word = target[i - 1];
    const WordId& curr_word = target[i]; 
    Expression prev_state = output_model->GetState();
    Expression context = attention_model->GetContext(encodings, prev_state);
    Expression new_state = output_model->AddInput(prev_word, context);
    word_losses[i] = output_model->Loss(new_state, curr_word);
  }
  return sum(word_losses);
}

vector<Sentence> Translator::Sample(const TranslatorInput* const source, unsigned sample_count, WordId BOS, WordId EOS, unsigned max_length) {
  ComputationGraph* cg = new ComputationGraph();
  vector<Sentence> samples;
  vector<Expression> encodings;
  for (unsigned i = 0; i < sample_count; ++i) {
    if (i % 10 == 0) {
      delete cg;
      cg = new ComputationGraph();
      NewGraph(*cg);
      encodings = encoder_model->Encode(source);
    }
    else {
      output_model->NewGraph(*cg);
    }

    WordId prev_word = BOS;
    Sentence sample;
    while (sample.size() < max_length) {
      Expression prev_state = output_model->GetState();
      Expression context = attention_model->GetContext(encodings, prev_state);
      Expression new_state = output_model->AddInput(prev_word, context);
      WordId word = output_model->Sample(new_state);
      prev_word = word;
      sample.push_back(word);
      if (word == EOS) {
        break;
      }
    }
    samples.push_back(sample);
  }
  delete cg;
  return samples;
}

vector<vector<float>> Translator::Align(const TranslatorInput* const source, const Sentence& target) {
  ComputationGraph cg;
  NewGraph(cg);
  vector<vector<float>> alignment;
  vector<Expression> encodings = encoder_model->Encode(source);
  Expression input_matrix = concatenate_cols(encodings);
  for (unsigned i = 1; i < target.size(); ++i) {
    const WordId& prev_word = target[i - 1];
    Expression state = output_model->GetState();
    Expression word_alignment = attention_model->GetAlignmentVector(encodings, state);
    Expression context = input_matrix * word_alignment;
    output_model->AddInput(prev_word, context);
    cg.incremental_forward();
    alignment.push_back(as_vector(word_alignment.value()));
  }
  return alignment;
}

KBestList<Sentence> Translator::Translate(const TranslatorInput* const source, unsigned K, unsigned beam_size, WordId BOS, WordId EOS, unsigned max_length) {
  assert (beam_size >= K);
  ComputationGraph cg;
  NewGraph(cg);

  KBestList<Sentence> complete_hyps(K);
  KBestList<pair<Sentence, RNNPointer>> top_hyps(beam_size);
  top_hyps.add(0.0, make_pair(Sentence(), output_model->GetStatePointer()));

  vector<Expression> encodings = encoder_model->Encode(source);
  for (unsigned length = 0; length < max_length; ++length) {
    KBestList<pair<Sentence, RNNPointer>> new_hyps(beam_size);

    for (auto& hyp : top_hyps.hypothesis_list()) {
      double hyp_score = get<0>(hyp);
      Sentence hyp_sentence = get<0>(get<1>(hyp));
      RNNPointer state_pointer = get<1>(get<1>(hyp));
      WordId prev_word = hyp_sentence.size() > 0 ? hyp_sentence.back() : BOS;
      assert (hyp_sentence.size() == length);

      Expression prev_state = output_model->GetState(state_pointer);
      Expression context = attention_model->GetContext(encodings, prev_state);
      Expression new_state = output_model->AddInput(prev_word, context, state_pointer);
      RNNPointer new_pointer = output_model->GetStatePointer();
      Expression dist_expr = output_model->PredictLogDistribution(new_state);
      vector<float> dist = as_vector(dist_expr.value());

      KBestList<WordId> best_words(beam_size);
      for (unsigned w = 0; w < dist.size(); ++w) {
        best_words.add(dist[w], w);
      }

      for (auto& w : best_words.hypothesis_list()) {
        double word_score = get<0>(w);
        WordId word = get<1>(w);
        double new_score = hyp_score + word_score;
        Sentence new_sentence = hyp_sentence;
        new_sentence.push_back(word);
        if (word != EOS) {
          new_hyps.add(new_score, make_pair(new_sentence, new_pointer));
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
    Sentence sentence = get<0>(get<1>(hyp));
    // TODO: Account for </s> in the score
    complete_hyps.add(score, sentence);
  }
  return complete_hyps;
}

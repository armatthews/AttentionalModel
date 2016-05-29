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

Expression Translator::BuildGraph(const Sentence* const source, const LinearSentence& target, ComputationGraph& cg) {
  NewGraph(cg);
  vector<Expression> word_losses(target.size());
  word_losses[0] = input(cg, 0.0f); // <s>

  vector<Expression> encodings;
  const SyntaxTree* tree = nullptr;
  bool SYNTAX_PRIOR = false; // XXX: Fix me! How do we know if we have a syntax prior? How does the syntax prior get access to the tree?
  if (SYNTAX_PRIOR) {
    tree = dynamic_cast<const SyntaxTree* const>(source);
    LinearSentence terminals = tree->GetTerminals();
    encodings = encoder_model->Encode(&terminals);
  }
  else {
    encodings = encoder_model->Encode(source);
  }

  attention_model->NewSentence(source);
  for (unsigned i = 1; i < target.size(); ++i) {
    const WordId& prev_word = target[i - 1];
    const WordId& curr_word = target[i]; 
    Expression prev_state = output_model->GetState();
    Expression context;
    if (SYNTAX_PRIOR) {
      context = dynamic_cast<StandardAttentionModel*>(attention_model)->GetContext(encodings, prev_state, tree);
    }
    else {
      context = dynamic_cast<StandardAttentionModel*>(attention_model)->GetContext(encodings, prev_state);
    }
    Expression new_state = output_model->AddInput(prev_word, context);
    word_losses[i] = output_model->Loss(new_state, curr_word);
  }
  return sum(word_losses);
}

void Translator::Sample(const vector<Expression>& encodings, LinearSentence& prefix, RNNPointer state_pointer, unsigned sample_count, WordId BOS, WordId EOS, unsigned max_length, ComputationGraph& cg, vector<LinearSentence>& samples) {
  WordId prev_word = prefix.size() > 0 ? prefix.back() : BOS;
  Expression prev_state = output_model->GetState(state_pointer);
  Expression context = attention_model->GetContext(encodings, prev_state);
  Expression new_state = output_model->AddInput(prev_word, context, state_pointer);
  RNNPointer new_pointer = output_model->GetStatePointer();

  unordered_map<WordId, unsigned> continuations;
  for (unsigned i = 0; i < sample_count; ++i) {
    WordId w = output_model->Sample(new_state);
    if (continuations.find(w) != continuations.end()) {
      continuations[w]++;
    }
    else {
      continuations[w] = 1;
    }
  }

  for (auto it = continuations.begin(); it != continuations.end(); ++it) {
    WordId w = it->first;
    prefix.push_back(w);
    if (w != EOS) {
      Sample(encodings, prefix, new_pointer, it->second, BOS, EOS, max_length - 1, cg, samples);
    }
    else {
      for (unsigned i = 0; i < it->second; ++i) {
        samples.push_back(prefix);
      }
    }
    prefix.pop_back();
  }
}

vector<LinearSentence> Translator::Sample(const Sentence* const source, unsigned sample_count, WordId BOS, WordId EOS, unsigned max_length) {
  ComputationGraph cg;
  NewGraph(cg);
  vector<Expression> encodings = encoder_model->Encode(source);
  LinearSentence prefix;
  vector<LinearSentence> samples;
  Sample(encodings, prefix, output_model->GetStatePointer(), sample_count, BOS, EOS, max_length, cg, samples);
  return samples;
}

vector<Expression> Translator::Align(const Sentence* const source, const LinearSentence& target, ComputationGraph& cg) {
  NewGraph(cg);
  vector<Expression> encodings = encoder_model->Encode(source);
  vector<Expression> alignments;
  Expression input_matrix = concatenate_cols(encodings);
  for (unsigned i = 1; i < target.size(); ++i) {
    const WordId& prev_word = target[i - 1];
    Expression state = output_model->GetState();
    Expression word_alignment = attention_model->GetAlignmentVector(encodings, state);
    Expression context = input_matrix * word_alignment;
    output_model->AddInput(prev_word, context);
    alignments.push_back(word_alignment);
  }
  return alignments;
}

KBestList<LinearSentence> Translator::Translate(const Sentence* const source, unsigned K, unsigned beam_size, WordId BOS, WordId EOS, unsigned max_length) {
  assert (beam_size >= K);
  ComputationGraph cg;
  NewGraph(cg);

  KBestList<LinearSentence> complete_hyps(K);
  KBestList<pair<LinearSentence, RNNPointer>> top_hyps(beam_size);
  top_hyps.add(0.0, make_pair(LinearSentence(), output_model->GetStatePointer()));

  vector<Expression> encodings = encoder_model->Encode(source);
  for (unsigned length = 0; length < max_length; ++length) {
    KBestList<pair<LinearSentence, RNNPointer>> new_hyps(beam_size);

    for (auto& hyp : top_hyps.hypothesis_list()) {
      double hyp_score = get<0>(hyp);
      LinearSentence hyp_sentence = get<0>(get<1>(hyp));
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
        LinearSentence new_sentence = hyp_sentence;
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
    LinearSentence sentence = get<0>(get<1>(hyp));
    // TODO: Account for </s> in the score
    complete_hyps.add(score, sentence);
  }
  return complete_hyps;
}

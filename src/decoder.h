#pragma once
#include "translator.h"
#include "kbestlist.h"

struct DecoderState {
  vector<vector<Expression>> source_encodings;
  vector<Expression> output_states;

  explicit DecoderState(unsigned n);
};

class Decoder {
public:
  explicit Decoder(Translator* translator);
  explicit Decoder(const vector<Translator*>& translators);
  void SetParams(unsigned max_length, WordId kSOS, WordId kEOS);

  vector<LinearSentence> SampleTranslations(const Sentence* source, unsigned n) const;
  LinearSentence Translate(const Sentence* source, unsigned beam_size) const;
  KBestList<LinearSentence> TranslateKBest(const Sentence* source, unsigned K, unsigned beam_size) const;
  vector<vector<float>> Align(const Sentence* source, const LinearSentence& target) const;
  vector<cnn::real> Loss(const Sentence* source, const LinearSentence& target) const;

private:
  vector<Translator*> translators;
  unsigned max_length;
  WordId kSOS;
  WordId kEOS;
};

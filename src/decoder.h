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
  void SetParams(unsigned max_length, const Word* const kSOS, const Word* const kEOS);

  vector<OutputSentence*> SampleTranslations(const InputSentence* const source, unsigned n) const;
  OutputSentence* Translate(const InputSentence* const source, unsigned beam_size) const;
  KBestList<OutputSentence*> TranslateKBest(const InputSentence* const source, unsigned K, unsigned beam_size) const;
  vector<vector<float>> Align(const InputSentence* const source, const OutputSentence* const target) const;
  vector<dynet::real> Loss(const InputSentence* const source, const OutputSentence* const target) const;

private:
  vector<Translator*> translators;
  unsigned max_length;
  const Word* kSOS;
  const Word* kEOS;
};

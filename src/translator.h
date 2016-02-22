#pragma once
#include <boost/serialization/access.hpp>
#include "encoder.h"
#include "attention.h"
#include "output.h"
#include "kbestlist.h"

class Translator {
public:
  Translator();
  Translator(EncoderModel* encoder, AttentionModel* attention, OutputModel* output);
  bool IsT2S() const;

  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  Expression BuildGraph(const TranslatorInput* const source, const Sentence& target, ComputationGraph& cg);
  vector<Sentence> Sample(const TranslatorInput* const source, unsigned samples, WordId BOS, WordId EOS, unsigned max_length);
  vector<Expression> Align(const TranslatorInput* const source, const Sentence& target, ComputationGraph& cg);
  // This could be used if your loss is over alignment matrices, for example
  // Expression Align(const TranslatorInput* const source, const Sentence& target, ComputationGraph& cg);
  KBestList<Sentence> Translate(const TranslatorInput* const source, unsigned K, unsigned beam_size, WordId BOS, WordId EOS, unsigned max_length);

private:
  EncoderModel* encoder_model;
  AttentionModel* attention_model;
  OutputModel* output_model;

  void Sample(const vector<Expression>& encodings, Sentence& prefix, RNNPointer state_pointer, unsigned sample_count, WordId BOS, WordId EOS, unsigned max_length, ComputationGraph& cg, vector<Sentence>& samples);

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & encoder_model;
    ar & attention_model;
    ar & output_model;
  }
};

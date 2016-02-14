#pragma once
#include <boost/serialization/access.hpp>
#include "encoder.h"
#include "attention.h"
#include "output.h"

class Translator {
public:
  Translator();
  Translator(EncoderModel* encoder, AttentionModel* attention, OutputModel* output);
  bool IsT2S() const;

  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  Expression BuildGraph(const TranslatorInput* const source, const Sentence& target, ComputationGraph& cg);
  vector<Sentence> Sample(const TranslatorInput* const source, unsigned samples, WordId BOS, WordId EOS, unsigned max_length);
  vector<vector<float>> Align(const TranslatorInput* const source, const Sentence& target, ComputationGraph& cg);

private:
  EncoderModel* encoder_model;
  AttentionModel* attention_model;
  OutputModel* output_model;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & encoder_model;
    ar & attention_model;
    ar & output_model;
  }
};

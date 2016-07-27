#pragma once
#include <boost/serialization/access.hpp>
#include "encoder.h"
#include "attention.h"
#include "output.h"
#include "kbestlist.h"
#include "syntax_tree.h"

class Translator {
public:
  Translator();
  Translator(EncoderModel* encoder, AttentionModel* attention, OutputModel* output);

  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  Expression BuildGraph(const InputSentence* const source, const OutputSentence* const target, ComputationGraph& cg);
  vector<OutputSentence*> Sample(const InputSentence* const source, unsigned samples, unsigned max_length);
  vector<Expression> Align(const InputSentence* const source, const OutputSentence* const target, ComputationGraph& cg);

  // This could be used if your loss is over alignment matrices, for example
  // Expression Align(const InputSentence* const source, const OutputSentence* target, ComputationGraph& cg);
  KBestList<OutputSentence*> Translate(const InputSentence* const source, unsigned K, unsigned beam_size, unsigned max_length);

private:
  EncoderModel* encoder_model;
  AttentionModel* attention_model;
  OutputModel* output_model;

  void Sample(const vector<Expression>& encodings, OutputSentence* prefix, RNNPointer state_pointer, unsigned sample_count, unsigned max_length, ComputationGraph& cg, vector<OutputSentence*>& samples);

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & encoder_model;
    ar & attention_model;
    ar & output_model;
  }
};

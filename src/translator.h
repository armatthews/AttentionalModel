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
  bool IsT2S() const;

  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  Expression BuildGraph(const Sentence* const source, const LinearSentence& target, ComputationGraph& cg);
  vector<LinearSentence> Sample(const Sentence* const source, unsigned samples, WordId BOS, WordId EOS, unsigned max_length);
  vector<Expression> Align(const Sentence* const source, const LinearSentence& target, ComputationGraph& cg);
  // This could be used if your loss is over alignment matrices, for example
  // Expression Align(const Sentence* const source, const LinearSentence& target, ComputationGraph& cg);
  KBestList<LinearSentence> Translate(const Sentence* const source, unsigned K, unsigned beam_size, WordId BOS, WordId EOS, unsigned max_length);

private:
  EncoderModel* encoder_model;
  AttentionModel* attention_model;
  OutputModel* output_model;

  void Sample(const vector<Expression>& encodings, LinearSentence& prefix, RNNPointer state_pointer, unsigned sample_count, WordId BOS, WordId EOS, unsigned max_length, ComputationGraph& cg, vector<LinearSentence>& samples);

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & encoder_model;
    ar & attention_model;
    ar & output_model;
  }
};

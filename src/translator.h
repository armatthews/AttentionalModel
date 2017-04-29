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
  vector<Expression> PerWordLosses(const InputSentence* const source, const OutputSentence* const target, ComputationGraph& cg);
  Expression BuildGraph(const InputSentence* const source, const OutputSentence* const target, ComputationGraph& cg);
  vector<pair<shared_ptr<OutputSentence>, float>> Sample(const InputSentence* const source, unsigned samples, unsigned max_length);
  vector<Expression> Align(const InputSentence* const source, const OutputSentence* const target, ComputationGraph& cg);
  KBestList<shared_ptr<OutputSentence>> Translate(const InputSentence* const source, unsigned K, unsigned beam_size, unsigned max_length, float length_bonus=0.0f);

  // XXX: This should be temporary and is just for some qualitative digging stuff I'm doing
  Expression GetContexts(const InputSentence* const source, const vector<Expression>& new_embs);
  Expression BackTranslate(const InputSentence* const source, const vector<Expression>& new_embs);
  vector<vector<float>> GetAttentionGradients(const InputSentence* const source, const OutputSentence* const target, ComputationGraph& cg);
  Expression BuildPredictionGraph(const InputSentence* const source, const vector<Expression>& target_probs, ComputationGraph& cg, const OutputSentence* const target);

//private:
  EncoderModel* encoder_model;
  AttentionModel* attention_model;
  OutputModel* output_model;

  void Sample(const vector<Expression>& encodings, shared_ptr<OutputSentence> prefix, float prefix_score, RNNPointer state_pointer, unsigned sample_count, unsigned max_length, ComputationGraph& cg, vector<pair<shared_ptr<OutputSentence>, float>>& samples);

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & encoder_model;
    ar & attention_model;
    ar & output_model;
  }
};

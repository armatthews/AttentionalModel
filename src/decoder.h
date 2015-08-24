#pragma once
#include "attentional.h"

struct DecoderState {
  vector<vector<Expression>> model_annotations;
  vector<MLP> model_aligners;
  vector<MLP> model_final_mlps;
  vector<OutputState> model_output_states;
  vector<vector<vector<float>>> model_alignments;
};

class AttentionalDecoder {
public:
  explicit AttentionalDecoder(AttentionalModel* model);
  explicit AttentionalDecoder(const vector<AttentionalModel*>& models);
  void SetParams(unsigned max_length, WordId kSOS, WordId kEOS);

  vector<WordId> SampleTranslation(const vector<WordId>& source);
  vector<WordId> Translate(const vector<WordId>& source, unsigned beam_size);
  KBestList<vector<WordId>> TranslateKBest(const vector<WordId>& source, unsigned K, unsigned beam_size);
  vector<vector<float>> Align(const vector<WordId>& source, const vector<WordId>& target);
  vector<cnn::real> Loss(const vector<WordId>& source, const vector<WordId>& target);

  vector<WordId> SampleTranslation(const SyntaxTree& source);
  vector<WordId> Translate(const SyntaxTree& source, unsigned beam_size);
  KBestList<vector<WordId>> TranslateKBest(const SyntaxTree& source, unsigned K, unsigned beam_size);
  vector<vector<float>> Align(const SyntaxTree& source, const vector<WordId>& target);
  vector<cnn::real> Loss(const SyntaxTree& source, const vector<WordId>& target);

private:
  // Each of these methods is a generic version that takes a DecoderState rather than a source object.
  // This allows us to minimize code duplication by abstracting away whether the input was a sentence
  // or a source-side syntax tree.
  vector<WordId> SampleTranslation(DecoderState& ds, ComputationGraph& cg);
  KBestList<vector<WordId>> TranslateKBest(DecoderState& ds, unsigned K, unsigned beam_size, ComputationGraph& cg);
  vector<vector<float>> Align(DecoderState& ds, const vector<WordId>& target, ComputationGraph& cg);
  vector<cnn::real> Loss(DecoderState& ds, const vector<WordId>& target, ComputationGraph& cg);

  // Initialize() calls InitializeAnnotations() then InitializeGivenAnnotations(),
  // and returns a DecoderState() which can then be passed on to one of the above methods
  DecoderState Initialize(const vector<WordId>& source, ComputationGraph& cg);
  DecoderState Initialize(const SyntaxTree& source, ComputationGraph& cg);
  tuple<vector<vector<Expression>>, vector<Expression>> InitializeAnnotations(const vector<WordId>& source, ComputationGraph& cg);
  tuple<vector<vector<Expression>>, vector<Expression>> InitializeAnnotations(const SyntaxTree& source, ComputationGraph& cg);
  DecoderState InitializeGivenAnnotations(const vector<vector<Expression>>& model_annotations, const vector<Expression> model_zeroth_contexts, ComputationGraph& cg);

  vector<AttentionalModel*> models;
  unsigned max_length;
  WordId kSOS;
  WordId kEOS;
};

tuple<Dict, Dict, vector<Model*>, vector<AttentionalModel*>> LoadModels(const vector<string>& model_filenames);
tuple<vector<WordId>, vector<WordId>> ReadInputLine(const string& line, Dict& source_vocab, Dict& target_vocab);
tuple<SyntaxTree, vector<WordId>> ReadT2SInputLine(const string& line, Dict& source_vocab, Dict& target_vocab);


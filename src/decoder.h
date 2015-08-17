#pragma once
#include "attentional.h"

class AttentionalDecoder {
public:
  explicit AttentionalDecoder(AttentionalModel* model);
  explicit AttentionalDecoder(const vector<AttentionalModel*>& models);
  void SetParams(unsigned max_length, WordId kSOS, WordId kEOS);

  vector<WordId> SampleTranslation(const vector<WordId>& source);
  vector<WordId> Translate(const vector<WordId>& source, unsigned beam_size);
  KBestList<vector<WordId>> TranslateKBest(const vector<WordId>& source, unsigned K, unsigned beam_size);
  vector<vector<float>> Align(const vector<WordId>& source, const vector<WordId>& target);

private:
  vector<AttentionalModel*> models;
  unsigned max_length;
  WordId kSOS;
  WordId kEOS;
};

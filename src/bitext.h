#pragma once
#include <vector>
#include <string>
#include <random>
#include <memory>
#include <utility>
#include "cnn/dict.h"
#include "syntax_tree.h"
#include "utils.h"

using namespace std;
using namespace cnn;

class Bitext {
public:
  shared_ptr<Dict> source_vocab;
  shared_ptr<Dict> target_vocab;

  virtual unsigned size() const;
  virtual void Shuffle(mt19937& rndeng);
  virtual bool ReadCorpus(const string& filename);
protected:
  void InitializeVocabularies();
};

class S2SBitext : public Bitext {
public:
  typedef vector<WordId> Source;
  typedef vector<WordId> Target;
  typedef pair<Source, Target> SentencePair;

  S2SBitext(Bitext* parent = nullptr);
  unsigned size() const;
  void Shuffle(mt19937& rndeng);
  bool ReadCorpus(const string& filename);
  const SentencePair& GetDatum(unsigned i) const;

  vector<SentencePair> data;
};

class T2SBitext : public Bitext {
public:
  typedef SyntaxTree Source;
  typedef vector<WordId> Target;
  typedef pair<Source, Target> SentencePair;

  T2SBitext(Bitext* parent = nullptr);
  unsigned size() const;
  void Shuffle(mt19937& rndeng);
  bool ReadCorpus(const string& filename);
  const SentencePair& GetDatum(unsigned i) const;

  vector<SentencePair> data;
};

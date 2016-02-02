#pragma once
#include <string>
#include <random>
#include "cnn/dict.h"

using namespace std;
using namespace cnn;

class Bitext {
public:
  Bitext();

  virtual unsigned size() const;
  virtual void Shuffle(mt19937& rndeng);
  static Bitext* Read(const string& filename, Dict* source_vocab, Dict* target_vocab);
};

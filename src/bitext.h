#pragma once
#include <vector>
#include "cnn/dict.h"
#include "syntax_tree.h"
#include "utils.h"

using namespace std;
using namespace cnn;

struct Bitext {
  Bitext();
  vector<vector<WordId> > source_sentences;
  vector<vector<WordId> > target_sentences;
  Dict source_vocab;
  Dict target_vocab;

  unsigned size() const;
};

struct T2SBitext {
  vector<SyntaxTree> source_trees;
  vector<vector<WordId> > target_sentences;
  Dict source_vocab;
  Dict target_vocab;

  unsigned size() const;
};

bool ReadCorpus(string filename, Bitext& bitext, bool add_bos_eos);
bool ReadT2SCorpus(string filename, T2SBitext& bitext, bool add_bos_eos);

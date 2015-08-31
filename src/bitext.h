#pragma once
#include <vector>
#include "cnn/dict.h"
#include "syntax_tree.h"
#include "utils.h"

using namespace std;
using namespace cnn;

struct Bitext {
  Bitext();
  Bitext(Bitext* parent); // Tie vocabularies
  shared_ptr<Dict> source_vocab;
  shared_ptr<Dict> target_vocab;
  vector<SyntaxTree> source_trees;
  vector<vector<WordId> > source_sentences;
  vector<vector<WordId> > target_sentences;

  unsigned size() const;
private:
  bool has_parent;
};

bool ReadCorpus(string filename, Bitext& bitext, bool t2s);

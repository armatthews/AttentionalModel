#include <fstream>
#include "bitext.h"
#include "utils.h"

using namespace std;

Bitext::Bitext() {
  source_vocab.Convert("UNK");
  source_vocab.Convert("<s>");
  source_vocab.Convert("</s>");

  target_vocab.Convert("UNK");
  target_vocab.Convert("<s>");
  target_vocab.Convert("</s>");
}

unsigned Bitext::size() const {
  assert(source_sentences.size() == target_sentences.size());
  return source_sentences.size();
}

bool ReadCorpus(string filename, Bitext& bitext, bool add_bos_eos) {
  ifstream f(filename);
  if (!f.is_open()) {
    return false;
  }

  WordId sBOS, sEOS, tBOS, tEOS;
  if (add_bos_eos) {
    sBOS = bitext.source_vocab.Convert("<s>");
    sEOS = bitext.source_vocab.Convert("</s>");
    tBOS = bitext.target_vocab.Convert("<s>");
    tEOS = bitext.target_vocab.Convert("</s>");
  }

  for (string line; getline(f, line);) {
    vector<WordId> source;
    vector<WordId> target;
    if (add_bos_eos) {
      source.push_back(sBOS);
      target.push_back(tBOS);
    }
    ReadSentencePair(line, &source, &bitext.source_vocab, &target, &bitext.target_vocab);
    if (add_bos_eos) {
      source.push_back(sEOS);
      target.push_back(tEOS);
    }
    bitext.source_sentences.push_back(source);
    bitext.target_sentences.push_back(target);
  }
  return true;
}

unsigned T2SBitext::size() const {
  assert(source_trees.size() == target_sentences.size());
  return source_trees.size();
}

bool ReadT2SCorpus(string filename, T2SBitext& bitext, bool add_bos_eos) {
  ifstream f(filename);
  if (!f.is_open()) {
    return false;
  }

  WordId tBOS, tEOS;
  if (add_bos_eos) {
    tBOS = bitext.target_vocab.Convert("<s>");
    tEOS = bitext.target_vocab.Convert("</s>");
  }

  for (string line; getline(f, line);) {
    vector<string> parts = tokenize(line, "|||");
    assert (parts.size() == 2);
    SyntaxTree source_tree(strip(parts[0]), &bitext.source_vocab);
    source_tree.AssignNodeIds();
    vector<WordId> target = ReadSentence(strip(parts[1]), &bitext.target_vocab);
    if (add_bos_eos) {
      target.insert(target.begin(), tBOS);
      target.push_back(tEOS);
    }
    bitext.source_trees.push_back(source_tree);
    bitext.target_sentences.push_back(target);
  }
  return true;
}


#include <fstream>
#include "bitext.h"
#include "utils.h"

using namespace std;

Bitext::Bitext() : has_parent(false) {
  source_vocab = make_shared<Dict>();
  source_vocab->Convert("UNK");
  source_vocab->Convert("<s>");
  source_vocab->Convert("</s>");

  target_vocab = make_shared<Dict>();
  target_vocab->Convert("UNK");
  target_vocab->Convert("<s>");
  target_vocab->Convert("</s>");
}

Bitext::Bitext(Bitext* parent) : has_parent(true){
  assert (parent != nullptr);
  source_vocab = parent->source_vocab;
  target_vocab = parent->target_vocab;
}

unsigned Bitext::size() const {
  assert(source_sentences.size() == target_sentences.size() || source_trees.size() == target_sentences.size());
  return target_sentences.size();
}

bool ReadCorpus(string filename, Bitext& bitext, bool t2s) {
  const bool add_bos_eos = true;
  ifstream f(filename);
  if (!f.is_open()) {
    return false;
  }

  WordId sBOS, sEOS, tBOS, tEOS;
  if (add_bos_eos) {
    sBOS = bitext.source_vocab->Convert("<s>");
    sEOS = bitext.source_vocab->Convert("</s>");
    tBOS = bitext.target_vocab->Convert("<s>");
    tEOS = bitext.target_vocab->Convert("</s>");
  }

  for (string line; getline(f, line);) {
    vector<string> parts = tokenize(line, "|||");
    assert (parts.size() == 2);
    if (t2s) {
      SyntaxTree source_tree(strip(parts[0]), bitext.source_vocab.get());
      source_tree.AssignNodeIds();
      bitext.source_trees.push_back(source_tree);
    }
    else {
      vector<WordId> source = ReadSentence(strip(parts[0]), bitext.source_vocab.get());
      if (add_bos_eos) {
        source.insert(source.begin(), sBOS);
        source.push_back(sEOS);
      }
      bitext.source_sentences.push_back(source);
    }

    vector<WordId> target = ReadSentence(strip(parts[1]), bitext.target_vocab.get());
    if (add_bos_eos) {
      target.insert(target.begin(), tBOS);
      target.push_back(tEOS);
    } 
    bitext.target_sentences.push_back(target);
  }
  return true;
}

#include <fstream>
#include "bitext.h"
#include "utils.h"

using namespace std;

void Bitext::InitializeVocabularies() {
  source_vocab = make_shared<Dict>();
  source_vocab->Convert("UNK");
  source_vocab->Convert("<s>");
  source_vocab->Convert("</s>");

  target_vocab = make_shared<Dict>();
  target_vocab->Convert("UNK");
  target_vocab->Convert("<s>");
  target_vocab->Convert("</s>");
}

S2SBitext::S2SBitext(Bitext* parent) {
  if (parent == nullptr) {
    InitializeVocabularies();
  }
  else {
    source_vocab = parent->source_vocab;
    target_vocab = parent->target_vocab;
  }
}

unsigned S2SBitext::size() const {
  return data.size();
}

void S2SBitext::Shuffle(mt19937& rndeng) {
  shuffle(data.begin(), data.end(), rndeng);
}

bool S2SBitext::ReadCorpus(string filename) {
  ifstream f(filename);
  if (!f.is_open()) {
    return false;
  }

  WordId sBOS = source_vocab->Convert("<s>");
  WordId sEOS = source_vocab->Convert("</s>");
  WordId tBOS = target_vocab->Convert("<s>");
  WordId tEOS = target_vocab->Convert("</s>");

  for (string line; getline(f, line);) {
    vector<string> parts = tokenize(line, "|||");
    assert (parts.size() == 2);

    vector<WordId> source = ReadSentence(strip(parts[0]), source_vocab.get());
    source.insert(source.begin(), sBOS);
    source.push_back(sEOS);

    vector<WordId> target = ReadSentence(strip(parts[1]), target_vocab.get());
    target.insert(target.begin(), tBOS);
    target.push_back(tEOS);

    data.push_back(make_pair(source, target));
  }

  return true;
}

const S2SBitext::SentencePair& S2SBitext::GetDatum(unsigned i) const {
  assert (i < data.size());
  return data[i];
}

T2SBitext::T2SBitext(Bitext* parent) {
  if (parent == nullptr) {
    InitializeVocabularies();
  }
  else {
    source_vocab = parent->source_vocab;
    target_vocab = parent->target_vocab;
  }
}

unsigned T2SBitext::size() const {
  return data.size();
}

void T2SBitext::Shuffle(mt19937& rndeng) {
  shuffle(data.begin(), data.end(), rndeng);
}

bool T2SBitext::ReadCorpus(string filename) {
  ifstream f(filename);
  if (!f.is_open()) {
    return false;
  }

  WordId tBOS = target_vocab->Convert("<s>");
  WordId tEOS = target_vocab->Convert("</s>");

  for (string line; getline(f, line);) {
    vector<string> parts = tokenize(line, "|||");
    assert (parts.size() == 2);

    SyntaxTree source(strip(parts[0]), source_vocab.get());
    source.AssignNodeIds();

    vector<WordId> target = ReadSentence(strip(parts[1]), target_vocab.get());
    target.insert(target.begin(), tBOS);
    target.push_back(tEOS);

    data.push_back(make_pair(source, target));
  }

  return true;
}

const T2SBitext::SentencePair& T2SBitext::GetDatum(unsigned i) const {
  assert (i < data.size());
  return data[i];
}

#include <fstream>
#include "io.h"

LinearSentence* ReadSentence(const string& line, Dict& dict) {
  vector<string> words = tokenize(strip(line), " ");
  LinearSentence* r = new LinearSentence();
  r->push_back(dict.Convert("<s>"));
  for (const string& w : words) {
    r->push_back(dict.Convert(w));
  }
  r->push_back(dict.Convert("</s>"));
  return r;
}

Bitext* ReadBitext(const string& filename, Dict& source_vocab, Dict& target_vocab) {
  ifstream f(filename);
  if (!f.is_open()) {
    return nullptr;
  }

  Bitext* bitext = new Bitext();
  for (string line; getline(f, line);) {
    vector<string> parts = tokenize(line, "|||");
    assert (parts.size() == 2);

    LinearSentence* source = ReadSentence(strip(parts[0]), source_vocab);
    LinearSentence* target = ReadSentence(strip(parts[1]), target_vocab);
    bitext->push_back(make_pair(source, target));
  }

  cerr << "Read " << bitext->size() << " lines from " << filename << endl;
  return bitext;
}

Bitext* ReadT2SBitext(const string& filename, Dict& source_vocab, Dict& target_vocab, Dict& label_vocab) {
  ifstream f(filename);
  if (!f.is_open()) {
    return nullptr;
  }

  Bitext* bitext = new Bitext();
  for (string line; getline(f, line);) {
    vector<string> parts = tokenize(line, "|||");
    assert (parts.size() == 2);

    SyntaxTree* source = new SyntaxTree(strip(parts[0]), &source_vocab, &label_vocab);
    source->AssignNodeIds();
    LinearSentence* target = ReadSentence(strip(parts[1]), target_vocab);
    bitext->push_back(make_pair(source, target));
  }

  cerr << "Read " << bitext->size() << " lines from " << filename << endl;
  return bitext;
}

void Serialize(const vector<Dict*>& dicts, const Translator& translator, Model& cnn_model) {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {}
  fseek(stdout, 0, SEEK_SET);

  boost::archive::binary_oarchive oa(cout);
  oa & cnn_model;
  oa & dicts;
  oa & translator;
}

void Deserialize(const string& filename, vector<Dict*>& dicts, Translator& translator, Model& cnn_model) {
  ifstream f(filename);
  boost::archive::binary_iarchive ia(f);
  ia & cnn_model;
  ia & dicts;
  ia & translator;
  f.close();

  for (Dict* dict : dicts) {
    assert (dict->is_frozen());
  }
}


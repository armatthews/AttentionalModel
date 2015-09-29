#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <iostream>
#include <fstream>
#include "io.h"

tuple<Dict, Dict, Model*, AttentionalModel*> LoadModel(const string& model_filename) {
  Dict source_vocab;
  Dict target_vocab;
  ifstream model_file(model_filename);
  if (!model_file.is_open()) {
    cerr << "ERROR: Unable to open " << model_filename << endl;
    exit(1);
  }
  boost::archive::text_iarchive ia(model_file);

  ia & source_vocab;
  ia & target_vocab;
  source_vocab.Freeze();
  target_vocab.Freeze();

  Model* cnn_model = new Model();
  //AttentionalModel* attentional_model = new AttentionalModel(*cnn_model, source_vocab.size(), target_vocab.size());
  AttentionalModel* attentional_model = new AttentionalModel();

  ia & *attentional_model;
  attentional_model->InitializeParameters(*cnn_model, source_vocab.size(), target_vocab.size());

  ia & *cnn_model;

  return make_tuple(source_vocab, target_vocab, cnn_model, attentional_model);
}

tuple<Dict, Dict, vector<Model*>, vector<AttentionalModel*>> LoadModels(const vector<string>& model_filenames) {
  vector<Model*> cnn_models;
  vector<AttentionalModel*> attentional_models;
  // XXX: We just use the last set of dictionaries, assuming they're all the same
  Dict source_vocab;
  Dict target_vocab;
  Model* cnn_model = nullptr;
  AttentionalModel* attentional_model = nullptr;
  for (const string& model_filename : model_filenames) {
    tie(source_vocab, target_vocab, cnn_model, attentional_model) = LoadModel(model_filename);
    cnn_models.push_back(cnn_model);
    attentional_models.push_back(attentional_model);
  }
  return make_tuple(source_vocab, target_vocab, cnn_models, attentional_models);
}

tuple<vector<WordId>, vector<WordId>> ReadInputLine(const string& line, Dict& source_vocab, Dict& target_vocab) {
  vector<string> parts = tokenize(line, "|||");
  parts = strip(parts);
  assert (parts.size() == 1 || parts.size() == 2);

  WordId ksSOS = source_vocab.Convert("<s>");
  WordId ksEOS = source_vocab.Convert("</s>");
  WordId ktSOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");

  vector<WordId> source = ReadSentence(parts[0], &source_vocab);
  source.insert(source.begin(), ksSOS);
  source.insert(source.end(), ksEOS);

  vector<WordId> target;
  if (parts.size() > 1) {
    target = ReadSentence(parts[1], &target_vocab);
    target.insert(target.begin(), ktSOS);
    target.push_back(ktEOS);
  }

  cerr << "Read input:";
  for (WordId w: source) {
    cerr << " " << source_vocab.Convert(w);
  }
  cerr << " |||";
  for (WordId w: target) {
    cerr << " " << target_vocab.Convert(w);
  }
  cerr << endl;

  return make_tuple(source, target);
}

tuple<SyntaxTree, vector<WordId>> ReadT2SInputLine(const string& line, Dict& source_vocab, Dict& target_vocab) {
  vector<string> parts = tokenize(line, "|||");
  parts = strip(parts);
  assert (parts.size() == 1 || parts.size() == 2);

  WordId ktSOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");

  SyntaxTree source_tree(parts[0], &source_vocab);
  source_tree.AssignNodeIds();

  vector<WordId> target;
  if (parts.size() > 1) {
    target = ReadSentence(parts[1], &target_vocab);
    target.insert(target.begin(), ktSOS);
    target.push_back(ktEOS);
  }

  cerr << "Read tree input:";
  for (WordId w: source_tree.GetTerminals()) {
    cerr << " " << source_vocab.Convert(w);
  }
  cerr << " |||";
  for (WordId w: target) {
    cerr << " " << target_vocab.Convert(w);
  }
  cerr << endl;

  return make_tuple(source_tree, target);
}


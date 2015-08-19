#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <fstream>
#include <csignal>

#include "bitext.h"
#include "attentional.h"
#include "decoder.h"
#include "utils.h"

using namespace cnn;
using namespace std;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

void trim(vector<string>& tokens, bool removeEmpty) {
  for (unsigned i = 0; i < tokens.size(); ++i) {
    boost::algorithm::trim(tokens[i]);
    if (tokens[i].length() == 0 && removeEmpty) {
      tokens.erase(tokens.begin() + i);
      --i;
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: cat source.txt | " << argv[0] << " model" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);


  cnn::Initialize(argc, argv);
  Dict source_vocab;
  Dict target_vocab;
  vector<Model*> cnn_models(argc - 1);
  vector<AttentionalModel*> attentional_models(argc - 1);
  for (unsigned i = 0; i < argc - 1; ++i) {
    const string model_filename = argv[i + 1];
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

    Model* model = new Model();
    cnn_models[i] = model;
    attentional_models[i] = new AttentionalModel(*model, source_vocab.size(), target_vocab.size());

    ia & *attentional_models[i];
    ia & *model;
  }

  WordId ksBOS = source_vocab.Convert("<s>");
  WordId ksEOS = source_vocab.Convert("</s>");
  WordId ktBOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");

  AttentionalDecoder decoder(attentional_models);
  decoder.SetParams(100, ktBOS, ktEOS);

  string line;
  for (; getline(cin, line);) {
    vector<string> parts = tokenize(line, "|||");
    trim(parts, false);

    vector<string> source_tokens = tokenize(parts[0], " ");
    trim(source_tokens, true);

    vector<WordId> source(source_tokens.size() + 2);
    source[0] = ksBOS;
    for (unsigned i = 0; i < source_tokens.size(); ++i) {
      source[i + 1] = source_vocab.Convert(source_tokens[i]);
    }
    source[source_tokens.size() + 1] = ksEOS;

    vector<string> target_tokens = tokenize(parts[1], " ");
    trim(target_tokens, true);

    vector<WordId> target(target_tokens.size() + 2);
    target[0] = ktBOS;
    for (unsigned i = 0; i < target_tokens.size(); ++i) {
      target[i + 1] = target_vocab.Convert(target_tokens[i]);
    }
    target[target_tokens.size() + 1] = ktEOS;

    cout << boost::algorithm::join(source_tokens, " ") << endl;
    cout << boost::algorithm::join(target_tokens, " ") << endl;

    assert (source[0] == ksBOS);
    assert (source[source.size() - 1] == ksEOS);
    assert (target[0] == ktBOS);
    assert (target[target.size() - 1] == ktEOS);

    vector<vector<float> > alignment = decoder.Align(source, target);
    unsigned j = 0;
    for (vector<float> v : alignment) {
      for (unsigned i = 0; i < v.size(); ++i) {
        cout << (i == 0 ? "" : " ") << v[i];
      }
      cout << endl;
    }
    cout << endl;
  }

  return 0;
}

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

  vector<Model*> cnn_models(argc - 1);
  vector<AttentionalModel*> attentional_models(argc - 1);
  // XXX: We just use the last set of dictionaries, assuming they're all the same
  Dict source_vocab;
  Dict target_vocab;
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

  AttentionalDecoder decoder(attentional_models);

  //WordId ksSOS = source_vocab.Convert("<s>");
  //WordId ksEOS = source_vocab.Convert("</s>");
  WordId ktSOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");

  unsigned beam_size = 10;
  unsigned max_length = 100;
  unsigned kbest_size = 10; 
  decoder.SetParams(max_length, ktSOS, ktEOS);

  string line;
  unsigned sentence_number = 0;
  while(getline(cin, line)) {
    vector<string> parts = tokenize(line, "|||");
    trim(parts, false);

    SyntaxTree source_tree(parts[0], &source_vocab);
    source_tree.AssignNodeIds();

    vector<string> tokens;
    for (WordId w : source_tree.GetTerminals()) {
      tokens.push_back(source_vocab.Convert(w));
    }
    cerr << "Read source sentence: " << boost::algorithm::join(tokens, " ") << endl;
    if (parts.size() > 1) {
      vector<string> reference = tokenize(parts[1], " ");
      trim(reference, true);
      cerr << "  Read reference: " << boost::algorithm::join(reference, " ") << endl;
    }

    KBestList<vector<WordId> > kbest = decoder.TranslateKBest(source_tree, kbest_size, beam_size);
    for (auto& scored_hyp : kbest.hypothesis_list()) {
      double score = scored_hyp.first;
      vector<WordId> hyp = scored_hyp.second;
      vector<string> words(hyp.size());
      for (unsigned i = 0; i < hyp.size(); ++i) {
        words[i] = target_vocab.Convert(hyp[i]);
      }
      string translation = boost::algorithm::join(words, " ");
      cout << sentence_number << " ||| " << translation << " ||| " << score << endl;
    }

    sentence_number++;

    if (ctrlc_pressed) {
      break;
    }
  }

  return 0;
}

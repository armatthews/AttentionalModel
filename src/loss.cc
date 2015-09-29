#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <csignal>

#include "bitext.h"
#include "attentional.h"
#include "decoder.h"
#include "utils.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: cat source.txt | " << argv[0] << " model" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);

  po::options_description desc("description");
  desc.add_options()
  ("models", po::value<vector<string>>()->required()->composing(), "model file(s), as output by train")
  ("t2s", po::bool_switch()->default_value(false), "Treat input as trees rather than normal sentences") // XXX: Can't we infer this from the model somehow?
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("models", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  vector<string> model_filenames = vm["models"].as<vector<string>>();
  const bool t2s = vm["t2s"].as<bool>();

  cnn::Initialize(argc, argv);

  Dict source_vocab;
  Dict target_vocab;
  vector<Model*> cnn_models;
  vector<AttentionalModel*> attentional_models;
  tie(source_vocab, target_vocab, cnn_models, attentional_models) = LoadModels(model_filenames);

  assert (source_vocab.Contains("<s>"));
  assert (source_vocab.Contains("</s>"));
  WordId ktSOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");
  source_vocab.Freeze();
  target_vocab.Freeze();

  AttentionalDecoder decoder(attentional_models);
  decoder.SetParams(0, ktSOS, ktEOS);

  string line;
  unsigned sentence_number = 0;
  while(getline(cin, line)) {
    vector<string> parts = tokenize(line, "|||");
    parts = strip(parts);

    vector<cnn::real> losses;
    if (t2s) {
      SyntaxTree source_tree;
      vector<WordId> target;
      tie(source_tree, target) = ReadT2SInputLine(line, source_vocab, target_vocab);
      losses = decoder.Loss(source_tree, target);
    }
    else {
      vector<WordId> source;
      vector<WordId> target;
      tie(source, target) = ReadInputLine(line, source_vocab, target_vocab);
      losses = decoder.Loss(source, target);
    }

    cnn::real total_loss = 0;
    for (unsigned i = 0; i < losses.size(); ++i) {
      total_loss += losses[i];
    }
    cout << "Total perp: " << exp(total_loss/losses.size()) << "\tPerp per word: ";
    for (unsigned i = 0; i < losses.size(); ++i) {
      cout << (i == 0 ? "" : " ") << exp(losses[i]);
    }
    cout << endl;
    cout.flush();

    sentence_number++;

    if (ctrlc_pressed) {
      break;
    }
  }

  return 0;
}

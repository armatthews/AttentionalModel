#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/program_options.hpp>
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
  ("samples,n", po::value<unsigned>()->default_value(1), "Number of samples per sentence")
  ("max_length", po::value<unsigned>()->default_value(100), "Maximum length of output sentences")
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
  unsigned samples = vm["samples"].as<unsigned>();
  unsigned max_length = vm["max_length"].as<unsigned>();
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
  decoder.SetParams(max_length, ktSOS, ktEOS);

  string line;
  unsigned sentence_number = 0;
  while(getline(cin, line)) {
    vector<string> parts = tokenize(line, "|||");
    parts = strip(parts);

    vector<vector<WordId>> outputs;
    if (t2s) {
      SyntaxTree source_tree;
      vector<WordId> target;
      tie(source_tree, target) = ReadT2SInputLine(line, source_vocab, target_vocab);
      outputs = decoder.SampleTranslations(source_tree, samples);
    }
    else {
      vector<WordId> source;
      vector<WordId> target;
      tie(source, target) = ReadInputLine(line, source_vocab, target_vocab);
      outputs = decoder.SampleTranslations(source, samples);
    }

    for (const vector<WordId>& output : outputs) {
      vector<string> words;
      for (WordId w : output) {
        words.push_back(target_vocab.Convert(w));
      }
      cout << sentence_number << " ||| " << boost::algorithm::join(words, " ") << endl;
      if (ctrlc_pressed) {
        break;
      }
    }
    cout.flush();
    sentence_number++;

    if (ctrlc_pressed) {
      break;
    }
  }

  return 0;
}

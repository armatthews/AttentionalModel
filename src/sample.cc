#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <fstream>
#include <csignal>

#include "train.h"
#include "utils.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

vector<WordId> ReadInputLine(string line, Dict& vocab) {
  vector<string> words = tokenize(line, " ");
  vector<WordId> r(words.size());
  for (unsigned i = 0; i < words.size(); ++i) {
    r[i] = vocab.Convert(words[i]);
  }
  return r;
}

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);
  cnn::Initialize(argc, argv);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "model file, as output by train")
  ("samples,n", po::value<unsigned>()->default_value(1), "Number of samples per sentence")
  ("max_length", po::value<unsigned>()->default_value(100), "Maximum length of output sentences")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("model", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  string model_filename = vm["model"].as<string>();
  unsigned num_samples = vm["samples"].as<unsigned>();
  unsigned max_length = vm["max_length"].as<unsigned>();

  Dict source_vocab;
  Dict target_vocab;
  Model cnn_model;
  Translator<Sentence> translator;
  Deserialize(model_filename, {&source_vocab, &target_vocab}, translator, cnn_model);

  assert (source_vocab.Contains("<s>"));
  assert (source_vocab.Contains("</s>"));
  WordId ktBOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");
  source_vocab.Freeze();
  target_vocab.Freeze();

  string line;
  unsigned sentence_number = 0;
  while(getline(cin, line)) {
    vector<WordId> source = ReadInputLine(line, source_vocab);

    ComputationGraph cg;
    vector<Sentence> samples = translator.Sample(source, num_samples, ktBOS, ktEOS, max_length, cg);
    for (Sentence sample : samples) {
      vector<string> words;
      for (WordId w : sample) {
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

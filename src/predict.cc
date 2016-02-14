#include "cnn/cnn.h"

#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>

#include "io.h"
#include "kbestlist.h"
#include "utils.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "model file(s), as output by train")
  ("kbest_size", po::value<unsigned>()->default_value(10), "K-best list size")
  ("beam_size", po::value<unsigned>()->default_value(10), "Beam size")
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

  const string model_filename = vm["model"].as<string>();
  const unsigned beam_size = vm["beam_size"].as<unsigned>();
  const unsigned max_length = vm["max_length"].as<unsigned>();
  const unsigned kbest_size = vm["kbest_size"].as<unsigned>();

  Model cnn_model;
  Translator translator;
  vector<Dict*> dicts;
  Deserialize(model_filename, dicts, translator, cnn_model);

  Dict* source_vocab = dicts[0];
  Dict* target_vocab = dicts[1];
  Dict* label_vocab = translator.IsT2S() ? dicts[2] : nullptr;

  assert (source_vocab->Contains("<s>"));
  assert (source_vocab->Contains("</s>"));
  WordId ktSOS = target_vocab->Convert("<s>");
  WordId ktEOS = target_vocab->Convert("</s>");
  assert (source_vocab->is_frozen());
  assert (target_vocab->is_frozen());

  string line;
  unsigned sentence_number = 0;
  while(getline(cin, line)) {
    TranslatorInput* source;
    if (translator.IsT2S()) {
      source = new SyntaxTree(line, source_vocab, label_vocab);
    }
    else {
      source = ReadSentence(line, *target_vocab);
    }

    KBestList<Sentence> kbest = translator.TranslateKBest(source, kbest_size, beam_size, max_length, ktSOS, ktEOS);
    OutputKBestList(sentence_number, kbest, *target_vocab);
    sentence_number++;
  }

  return 0;
}

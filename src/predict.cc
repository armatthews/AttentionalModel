#include "dynet/dynet.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <fstream>

#include "io.h"
#include "kbestlist.h"
#include "utils.h"

using namespace dynet;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "model file(s), as output by train")
  ("input_source", po::value<string>()->required(), "input file source")
  ("kbest_size,k", po::value<unsigned>()->default_value(1), "K-best list size")
  ("beam_size,b", po::value<unsigned>()->default_value(10), "Beam size")
  ("max_length", po::value<unsigned>()->default_value(100), "Maximum length of output sentences")
  ("length_bonus", po::value<float>()->default_value(0.0f), "Length bonus per word")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("model", 1);
  positional_options.add("input_source", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const string model_filename = vm["model"].as<string>();
  const string input_source = vm["input_source"].as<string>();
  const unsigned beam_size = vm["beam_size"].as<unsigned>();
  const unsigned max_length = vm["max_length"].as<unsigned>();
  const unsigned kbest_size = vm["kbest_size"].as<unsigned>();
  const float length_bonus = vm["length_bonus"].as<float>();

  InputReader* input_reader = nullptr;
  OutputReader* output_reader = nullptr;
  Model dynet_model;
  Translator translator;
  Trainer* trainer = nullptr;
  Deserialize(model_filename, input_reader, output_reader, translator, dynet_model, trainer);
  translator.SetDropout(0.0f);

  vector<InputSentence*> source_sentences = input_reader->Read(input_source);
  for (unsigned sentence_number = 0; sentence_number < source_sentences.size(); ++sentence_number) {
    InputSentence* source = source_sentences[sentence_number];

    KBestList<shared_ptr<OutputSentence>> kbest = translator.Translate(source, kbest_size, beam_size, max_length, length_bonus);
    OutputKBestList(sentence_number, kbest, output_reader);

    cout.flush();
  }

  return 0;
}

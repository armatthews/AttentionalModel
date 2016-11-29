#include "dynet/dynet.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <fstream>

#include "io.h"
#include "utils.h"

using namespace dynet;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "model file, as output by train")
  ("input_source", po::value<string>()->required(), "input file source")
  ("samples,n", po::value<unsigned>()->default_value(1), "Number of samples per sentence")
  ("max_length", po::value<unsigned>()->default_value(100), "Maximum length of output sentences")
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

  string model_filename = vm["model"].as<string>();
  unsigned num_samples = vm["samples"].as<unsigned>();
  unsigned max_length = vm["max_length"].as<unsigned>();

  InputReader* input_reader = nullptr;
  OutputReader* output_reader = nullptr;
  Translator translator;
  Model dynet_model;
  Trainer* trainer = nullptr;
  Deserialize(model_filename, input_reader, output_reader, translator, dynet_model, trainer);
  translator.SetDropout(0.0f);

  vector<InputSentence*> source_sentences = input_reader->Read(vm["input_source"].as<string>());
  for (unsigned sentence_number = 0; sentence_number < source_sentences.size(); ++sentence_number) {
    InputSentence* source = source_sentences[sentence_number];

    vector<pair<shared_ptr<OutputSentence>, float>> samples = translator.Sample(source, num_samples, max_length);
    for (auto scored_sample : samples) {
      auto& sample = get<0>(scored_sample);
      float score = get<1>(scored_sample);
      vector<string> words;
      for (Word* w : *sample) {
        words.push_back(output_reader->ToString(w));
      }
      cout << sentence_number << " ||| " << boost::algorithm::join(words, " ") << " ||| " << score << endl;
    }
    cout.flush();
  }

  return 0;
}

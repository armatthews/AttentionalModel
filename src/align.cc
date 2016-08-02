#include "cnn/cnn.h"

#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>

#include "io.h"
#include "utils.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv) {
  cnn::initialize(argc, argv);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "model file, as output by train")
  ("input_source", po::value<string>()->required(), "input file source")
  ("input_target", po::value<string>()->required(), "input file target")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("model", 1);
  positional_options.add("input_source", 1);
  positional_options.add("input_target", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const string model_filename = vm["model"].as<string>();

  InputReader* input_reader = nullptr;
  OutputReader* output_reader = nullptr;
  Translator translator;
  Model cnn_model;
  Trainer* trainer = nullptr;
  Deserialize(model_filename, input_reader, output_reader, translator, cnn_model, trainer);
  translator.SetDropout(0.0f);

  vector<InputSentence*> source_sentences = input_reader->Read(vm["input_source"].as<string>());
  vector<OutputSentence*> target_sentences = output_reader->Read(vm["input_target"].as<string>());
  assert (source_sentences.size() == target_sentences.size());
  for (unsigned sentence_number = 0; sentence_number < source_sentences.size(); ++sentence_number) {
    InputSentence* source = source_sentences[sentence_number];
    OutputSentence* target = target_sentences[sentence_number];

    ComputationGraph cg;
    vector<Expression> alignment = translator.Align(source, target, cg);
    cg.incremental_forward();

    assert (alignment.size() > 0);

    unsigned j = 0;
    for (Expression a : alignment) {
      vector<float> v = as_vector(a.value());
      for (unsigned i = 0; i < v.size(); ++i) {
        cout << (i == 0 ? "" : " ") << v[i];
      }
      cout << endl;
      ++j;
    }
    cout << endl;
    cout.flush();

    sentence_number++;
  }

  return 0;
}

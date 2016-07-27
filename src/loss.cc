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
  ("model", po::value<string>()->required(), "model files, as output by train")
  ("input_source", po::value<string>()->required(), "input file source")
  ("input_target", po::value<string>()->required(), "input file target")
  ("perp", "Show per-sentence perplexity instead of negative log prob")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("model", 1);
  positional_options.add("input_source", 1);
  positional_options.add("input_target", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm, true);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const string model_filename = vm["model"].as<string>();
  const bool show_perp = vm.count("perp") > 0;

  InputReader* input_reader = nullptr;
  OutputReader* output_reader = nullptr;
  Translator translator;
  Model cnn_model;
  Deserialize(model_filename, input_reader, output_reader, translator, cnn_model);
  translator.SetDropout(0.0f);

  cnn::real total_loss = 0;
  unsigned total_words = 0;
  vector<InputSentence*> source_sentences = input_reader->Read(vm["input_source"].as<string>());
  vector<OutputSentence*> target_sentences = output_reader->Read(vm["input_target"].as<string>());
  assert (source_sentences.size() == target_sentences.size());
  for (unsigned sentence_number = 0; sentence_number < source_sentences.size(); ++sentence_number) {
    InputSentence* source = source_sentences[sentence_number];
    OutputSentence* target = target_sentences[sentence_number];

    ComputationGraph cg;
    Expression loss_expr = translator.BuildGraph(source, target, cg);
    cg.incremental_forward();
    cnn::real loss = as_scalar(loss_expr.value());
    unsigned words = target->size();
    if (show_perp) {
      cout << sentence_number << " ||| " << exp(loss / words) << endl;
    }
    else {
      cout << sentence_number << " ||| " << loss << endl;
    }
    cout.flush();

    total_loss += loss;
    total_words += words;
  }

  if (show_perp) {
    cout << "Total ||| " << exp(total_loss / total_words) << endl;
  }
  else {
    cout << "Total ||| " << total_loss << endl;
  }

  return 0;
}

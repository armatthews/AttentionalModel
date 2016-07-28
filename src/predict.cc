#include "cnn/cnn.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <fstream>

#include "io.h"
#include "kbestlist.h"
#include "utils.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

void OutputKBestList(unsigned sentence_number, KBestList<OutputSentence*> kbest, OutputReader* output_reader) {
  for (auto& scored_hyp : kbest.hypothesis_list()) {
    double score = scored_hyp.first;
    const OutputSentence* const hyp = scored_hyp.second;
    vector<string> words(hyp->size());
    for (unsigned i = 0; i < hyp->size(); ++i) {
      words[i] = output_reader->ToString(hyp->at(i));
    }
    string translation = boost::algorithm::join(words, " ");
    cout << sentence_number << " ||| " << translation << " ||| " << score << endl;
  }
  cout.flush();
}

int main(int argc, char** argv) {
  cnn::initialize(argc, argv);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "model file(s), as output by train")
  ("input_source", po::value<string>()->required(), "input file source")
  ("kbest_size", po::value<unsigned>()->default_value(10), "K-best list size")
  ("beam_size", po::value<unsigned>()->default_value(10), "Beam size")
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

  const string model_filename = vm["model"].as<string>();
  const unsigned beam_size = vm["beam_size"].as<unsigned>();
  const unsigned max_length = vm["max_length"].as<unsigned>();
  const unsigned kbest_size = vm["kbest_size"].as<unsigned>();

  InputReader* input_reader = nullptr;
  OutputReader* output_reader = nullptr;
  Translator translator;
  Model cnn_model;
  Deserialize(model_filename, input_reader, output_reader, translator, cnn_model);

  vector<InputSentence*> source_sentences = input_reader->Read(vm["input_source"].as<string>());

  for (unsigned sentence_number = 0; sentence_number < source_sentences.size(); ++sentence_number) {
    InputSentence* source = source_sentences[sentence_number];
    KBestList<OutputSentence*> kbest = translator.Translate(source, kbest_size, beam_size, max_length);
    OutputKBestList(sentence_number, kbest, output_reader);
  }

  return 0;
}

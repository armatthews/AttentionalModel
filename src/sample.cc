#include "cnn/cnn.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <fstream>

#include "io.h"
#include "utils.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv) {  
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

  Model cnn_model;
  Translator translator;
  vector<Dict*> dicts;
  Deserialize(model_filename, dicts, translator, cnn_model);

  for (Dict* dict : dicts) {
    assert (dict->is_frozen());
  }

  Dict* source_vocab = dicts[0];
  Dict* target_vocab = dicts[1];
  Dict* label_vocab = translator.IsT2S() ? dicts[2] : nullptr;

  source_vocab->Freeze();
  target_vocab->Freeze();

  assert (source_vocab->Contains("<s>"));
  assert (source_vocab->Contains("</s>"));
  WordId ktBOS = target_vocab->Convert("<s>");
  WordId ktEOS = target_vocab->Convert("</s>");

  string line;
  unsigned sentence_number = 0;
  while(getline(cin, line)) {
    cerr << line << endl;
    TranslatorInput* source;
    if (translator.IsT2S()) {
      SyntaxTree* source_tree = new SyntaxTree(line, source_vocab, label_vocab);
      source_tree->AssignNodeIds();
      source = source_tree;
    }
    else {
      source = ReadSentence(line, *source_vocab);
    }

    vector<Sentence> samples = translator.Sample(source, num_samples, ktBOS, ktEOS, max_length);
    for (Sentence sample : samples) {
      vector<string> words;
      for (WordId w : sample) {
        words.push_back(target_vocab->Convert(w));
      }
      cout << sentence_number << " ||| " << boost::algorithm::join(words, " ") << endl;
    }
    cout.flush();
    sentence_number++;
  }

  return 0;
}

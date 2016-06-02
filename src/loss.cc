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
  cnn::Initialize(argc, argv);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "model files, as output by train")
  ("perp", "Show per-sentence perplexity instead of negative log prob")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("model", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm, true);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const string model_filename = vm["model"].as<string>();
  const bool show_perp = vm.count("perp") > 0;

  Model cnn_model;
  Translator translator;
  vector<Dict*> dicts;
  Deserialize(model_filename, dicts, translator, cnn_model);
  translator.SetDropout(0.0f);

  Dict* source_vocab = dicts[0];
  Dict* target_vocab = dicts[1];
  Dict* label_vocab = translator.IsT2S() ? dicts[2] :  nullptr;

  string line;
  unsigned sentence_number = 0;
  cnn::real total_loss = 0;
  unsigned total_words = 0;
  while(getline(cin, line)) {
    vector<string> parts = tokenize(line, "|||");
    parts = strip(parts);

    vector<cnn::real> losses;
    InputSentence* source;
    if (translator.IsT2S()) {
      source = new SyntaxTree(parts[0], source_vocab, label_vocab);
    }
    else {
      source = ReadSentence(parts[0], *source_vocab);
    }
    OutputSentence* target = ReadSentence(parts[1], *target_vocab); 

    ComputationGraph cg;
    Expression loss_expr = translator.BuildGraph(source, target, cg);
    cg.incremental_forward();
    cnn::real loss = as_scalar(loss_expr.value());
    unsigned words = target->size() - 1;
    if (show_perp) {
      cout << sentence_number << " ||| " << exp(loss / words) << endl;
    }
    else {
      cout << sentence_number << " ||| " << loss << endl;
    }
    cout.flush();

    sentence_number++;
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

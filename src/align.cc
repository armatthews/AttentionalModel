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
  ("model", po::value<string>()->required(), "model file, as output by train")
  ("show_eos", po::bool_switch()->default_value(false), "Show alignment links for the target word </s>")
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
  const bool show_eos = vm["show_eos"].as<bool>();

  Model cnn_model;
  Translator translator;
  vector<Dict*> dicts;
  Deserialize(model_filename, dicts, translator, cnn_model);

  Dict* source_vocab = dicts[0];
  Dict* target_vocab = dicts[1];
  Dict* label_vocab = translator.IsT2S() ? dicts[2] : nullptr;

  assert (source_vocab->Contains("<s>"));
  assert (source_vocab->Contains("</s>"));
  assert (target_vocab->Contains("<s>"));
  assert (target_vocab->Contains("</s>"));

  string line;
  unsigned sentence_number = 0;
  while(getline(cin, line)) {
    cerr << line << endl;
    vector<string> parts = tokenize(line, "|||");
    parts = strip(parts);

    TranslatorInput* source;
    if (translator.IsT2S()) {
      source = new SyntaxTree(parts[0], source_vocab, label_vocab);
    }
    else {
      source = ReadSentence(parts[0], *source_vocab);
    }
    Sentence* target = ReadSentence(parts[1], *target_vocab); 

    ComputationGraph cg;
    vector<Expression> alignment = translator.Align(source, *target, cg);
    cg.incremental_forward();

    assert (alignment.size() > 0);
    if (!show_eos) {
      alignment.pop_back();
    }

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

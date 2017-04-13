#include "dynet/dynet.h"
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>

#include "io.h"
#include "utils.h"

using namespace dynet;
using namespace std;
namespace po = boost::program_options;

ostream& operator<<(ostream& os, const vector<float>& vals) {
  for (unsigned i = 0; i < vals.size(); ++i) {
    os << (i == 0 ? "" : " ") << vals[i];
  }
  return os;
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "model files, as output by train")
  ("input_source", po::value<string>()->required(), "input file source")
  ("input_target", po::value<string>()->required(), "input file target")
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

  InputReader* input_reader = nullptr;
  OutputReader* output_reader = nullptr;
  Translator translator;
  Model dynet_model;
  Trainer* trainer = nullptr;
  Deserialize(model_filename, input_reader, output_reader, translator, dynet_model, trainer);

  const string input_source = vm["input_source"].as<string>();
  const string input_target = vm["input_target"].as<string>();
  Bitext bitext = ReadBitext(input_source, input_target, input_reader, output_reader);

  for (unsigned i = 0; i < bitext.size(); ++i) {
    ComputationGraph cg;
    InputSentence* source = get<0>(bitext[i]);
    OutputSentence* target = get<1>(bitext[i]);

    translator.NewGraph(cg);
    vector<vector<float>> grads = translator.GetAttentionGradients(source, target, cg);
    for (unsigned j = 0; j < grads.size(); ++j) {
      //cout << i << "\t" << j << "\t";
      for (unsigned k = 0; k < grads[j].size(); ++k) {
        cout << (k == 0 ? "" : " ") << grads[j][k];
      }
      cout << endl;
    }
    cout << endl;
  }
  return 0;
}

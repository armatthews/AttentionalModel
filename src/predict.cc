#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <csignal>

#include "bitext.h"
#include "attentional.h"
#include "decoder.h"
#include "utils.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

tuple<Dict, Dict, vector<Model*>, vector<AttentionalModel*>> LoadModels(const vector<string>& model_filenames) {
  vector<Model*> cnn_models;
  vector<AttentionalModel*> attentional_models;
  // XXX: We just use the last set of dictionaries, assuming they're all the same
  Dict source_vocab;
  Dict target_vocab;
  for (const string& model_filename : model_filenames) {
    ifstream model_file(model_filename);
    if (!model_file.is_open()) {
      cerr << "ERROR: Unable to open " << model_filename << endl;
      exit(1);
    }
    boost::archive::text_iarchive ia(model_file);

    ia & source_vocab;
    ia & target_vocab;
    source_vocab.Freeze();
    target_vocab.Freeze();

    Model* model = new Model();
    AttentionalModel* attentional_model = new AttentionalModel(*model, source_vocab.size(), target_vocab.size());
    cnn_models.push_back(model);
    attentional_models.push_back(attentional_model);

    ia & *attentional_model;
    ia & *model;
  }
  return make_tuple(source_vocab, target_vocab, cnn_models, attentional_models);
}

void OutputKBestList(unsigned sentence_number, KBestList<vector<WordId>> kbest, Dict& target_vocab) {
  for (auto& scored_hyp : kbest.hypothesis_list()) {
    double score = scored_hyp.first;
    vector<WordId> hyp = scored_hyp.second;
    vector<string> words(hyp.size());
    for (unsigned i = 0; i < hyp.size(); ++i) {
      words[i] = target_vocab.Convert(hyp[i]);
    }
    string translation = boost::algorithm::join(words, " ");
    cout << sentence_number << " ||| " << translation << " ||| " << score << endl;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: cat source.txt | " << argv[0] << " model" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);

  po::options_description desc("description");
  desc.add_options()
  ("models", po::value<vector<string>>()->required()->composing(), "model file(s), as output by train ")
  ("kbest_size", po::value<unsigned>()->default_value(10), "K-best list size")
  ("beam_size", po::value<unsigned>()->default_value(10), "Beam size")
  ("max_length", po::value<unsigned>()->default_value(100), "Maximum length of output sentences")
  ("t2s", po::bool_switch()->default_value(false), "Treat input as trees rather than normal sentences") // XXX: Can't we infer this from the model somehow?
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("models", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  vector<string> model_filenames = vm["models"].as<vector<string>>();
  const unsigned beam_size = vm["beam_size"].as<unsigned>();
  const unsigned max_length = vm["max_length"].as<unsigned>();
  const unsigned kbest_size = vm["kbest_size"].as<unsigned>();
  const bool t2s = vm["t2s"].as<bool>();

  cnn::Initialize(argc, argv);

  Dict source_vocab;
  Dict target_vocab;
  vector<Model*> cnn_models;
  vector<AttentionalModel*> attentional_models;
  tie(source_vocab, target_vocab, cnn_models, attentional_models) = LoadModels(model_filenames);

  assert (source_vocab.Contains("<s>"));
  assert (source_vocab.Contains("</s>"));
  WordId ksSOS = source_vocab.Convert("<s>");
  WordId ksEOS = source_vocab.Convert("</s>");
  WordId ktSOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");
  source_vocab.Freeze();
  target_vocab.Freeze();

  AttentionalDecoder decoder(attentional_models);
  decoder.SetParams(max_length, ktSOS, ktEOS);

  string line;
  unsigned sentence_number = 0;
  while(getline(cin, line)) {
    vector<string> parts = tokenize(line, "|||");
    parts = strip(parts);

    if (t2s) {
      SyntaxTree source_tree(parts[0], &source_vocab);
      source_tree.AssignNodeIds();

      vector<string> tokens;
      for (WordId w : source_tree.GetTerminals()) {
        tokens.push_back(source_vocab.Convert(w));
      }
      cerr << "Read source sentence: " << boost::algorithm::join(tokens, " ") << endl;

      if (parts.size() > 1) {
        vector<string> reference = tokenize(parts[1], " ");
        reference = strip(reference, true);
        cerr << "Read reference: " << boost::algorithm::join(reference, " ") << endl;
      }

      KBestList<vector<WordId> > kbest = decoder.TranslateKBest(source_tree, kbest_size, beam_size);
      OutputKBestList(sentence_number, kbest, target_vocab);
    }
    else {
      vector<string> tokens = tokenize(parts[0], " ");
      tokens = strip(tokens, true);

      vector<WordId> source(tokens.size());
      for (unsigned i = 0; i < tokens.size(); ++i) {
        source[i] = source_vocab.Convert(tokens[i]);
      }
      source.insert(source.begin(), ksSOS);
      source.insert(source.end(), ksEOS);
      cerr << "Read source sentence: " << boost::algorithm::join(tokens, " ") << endl;

      if (parts.size() > 1) {
        vector<string> reference = tokenize(parts[1], " ");
        reference = strip(reference, true);
        cerr << "Read reference: " << boost::algorithm::join(reference, " ") << endl;
      }

      KBestList<vector<WordId> > kbest = decoder.TranslateKBest(source, kbest_size, beam_size);
      OutputKBestList(sentence_number, kbest, target_vocab);
    }
    sentence_number++;

    if (ctrlc_pressed) {
      break;
    }
  }

  return 0;
}

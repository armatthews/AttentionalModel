#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <csignal>
#include <algorithm>

#include "bitext.h"
#include "attentional.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    cerr << "Exiting..." << endl;
    exit(1);
  }
  else {
    cerr << "Ctrl-c pressed!" << endl;
    ctrlc_pressed = true;
  }
}

template <class RNG>
void shuffle(Bitext& bitext, RNG& g) {
  vector<unsigned> indices(bitext.size(), 0);
  for (unsigned i = 0; i < bitext.size(); ++i) {
    indices[i] = i;
  }
  shuffle(indices.begin(), indices.end(), g);

  vector<vector<WordId>> source_sentences(bitext.source_sentences.size());
  vector<SyntaxTree> source_trees(bitext.source_trees.size());
  vector<vector<WordId> > target_sentences(bitext.target_sentences.size());
  for (unsigned i = 0; i < bitext.size(); ++i) {
    if (i < bitext.source_trees.size()) {
      source_trees[i] = bitext.source_trees[i];
    }
    if (i < bitext.source_sentences.size()) {
      source_sentences[i] = bitext.source_sentences[i];
    }
    target_sentences[i] = bitext.target_sentences[i];
  }
  bitext.source_trees = source_trees;
  bitext.source_sentences = source_sentences;
  bitext.target_sentences = target_sentences;
}

void Serialize(Bitext& bitext, AttentionalModel& attentional_model, Model& model) {
  int r = ftruncate(fileno(stdout), 0);
  fseek(stdout, 0, SEEK_SET);

  boost::archive::text_oarchive oa(cout);
  oa & *bitext.source_vocab;
  oa & *bitext.target_vocab;
  oa << attentional_model;
  oa << model;
}

pair<cnn::real, unsigned> ComputeLoss(const Bitext& bitext, AttentionalModel& attentional_model, bool t2s) {
  cnn::real loss = 0.0;
  unsigned word_count = 0;
  for (unsigned i = 0; i < bitext.size(); ++i) {
    ComputationGraph cg;
    const vector<WordId>& target_sentence = bitext.target_sentences[i];
    word_count += bitext.target_sentences[i].size() - 1; // Minus one for <s>
    if (t2s) {
      const SyntaxTree& source_tree = bitext.source_trees[i];
      attentional_model.BuildGraph(source_tree, target_sentence, cg);
    }
    else {
      const vector<WordId>& source_sentence = bitext.source_sentences[i];
      attentional_model.BuildGraph(source_sentence, target_sentence, cg);
    }
    double l = as_scalar(cg.forward());
    loss += l;
    if (ctrlc_pressed) {
      break;
    }
  }
  return make_pair(loss, word_count);
}

Bitext* ReadBitext(const string& filename, Bitext* parent, bool t2s) {
  Bitext* bitext = (parent == NULL) ? new Bitext() : new Bitext(parent);
  ReadCorpus(filename, *bitext, t2s);
  cerr << "Read " << bitext->size() << " lines from " << filename << endl;
  cerr << "Vocab size: " << bitext->source_vocab->size() << "/" << bitext->target_vocab->size() << endl;
  return bitext;
}

Bitext* ReadBitext(const string& filename, bool t2s) {
  return ReadBitext(filename, nullptr, t2s);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " corpus.txt" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);

  po::options_description desc("description");
  desc.add_options()
  ("train_bitext", po::value<string>()->required(), "Training bitext in source_tree ||| target format")
  ("dev_bitext", po::value<string>()->default_value(""), "(Optional) Dev bitext, used for early stopping")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("t2s", po::bool_switch()->default_value(false), "Treat input as trees rather than normal sentences")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("train_bitext", 1);
  positional_options.add("dev_bitext", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const string train_bitext_filename = vm["train_bitext"].as<string>();
  const string dev_bitext_filename = vm["dev_bitext"].as<string>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const bool t2s = vm["t2s"].as<bool>();

  Bitext* train_bitext = ReadBitext(train_bitext_filename, t2s);
  Bitext* dev_bitext = ReadBitext(dev_bitext_filename, train_bitext, t2s);

  cnn::Initialize(argc, argv);
  std::mt19937 rndeng(42);
  Model model;
  AttentionalModel attentional_model(model, train_bitext->source_vocab->size(), train_bitext->target_vocab->size());
  //SimpleSGDTrainer sgd(&model, 0.0, 0.1);
  //AdagradTrainer sgd(&model, 0.0, 0.1);
  //AdadeltaTrainer sgd(&model, 0.0);
  //AdadeltaTrainer sgd(&model, 0.0, 1e-6, 0.992);
  //RmsPropTrainer sgd(&model, 0.0, 0.1);
  AdamTrainer sgd(&model, 1.e-4);
  sgd.eta_decay = 0.05;

  cerr << "Training model...\n";
  unsigned minibatch_count = 0;
  const unsigned minibatch_size = 10;
  cnn::real best_dev_loss = numeric_limits<cnn::real>::max();
  for (unsigned iteration = 0; iteration < num_iterations; iteration++) {
    unsigned word_count = 0;
    unsigned tword_count = 0;
    shuffle(*train_bitext, rndeng);
    double loss = 0.0;
    double tloss = 0.0;
    for (unsigned i = 0; i < train_bitext->size(); ++i) {
      ComputationGraph cg;
      const vector<WordId>& target_sentence = train_bitext->target_sentences[i];
      word_count += train_bitext->target_sentences[i].size() - 1; // Minus one for <s>
      tword_count += train_bitext->target_sentences[i].size() - 1; // Minus one for <s>
      if (t2s) {
        const SyntaxTree& source_tree = train_bitext->source_trees[i];
        attentional_model.BuildGraph(source_tree, target_sentence, cg);
      }
      else {
        const vector<WordId>& source_sentence = train_bitext->source_sentences[i];
        attentional_model.BuildGraph(source_sentence, target_sentence, cg);
      }
      cg.forward();
      double l = as_scalar(cg.forward());
      loss += l;
      tloss += l;
      cg.backward();
      if (i % 50 == 0 && i > 0) {
        float fractional_iteration = (float)iteration + ((float)i / train_bitext->size());
        cerr << "--" << fractional_iteration << " loss: " << tloss << " (perp=" << exp(tloss/tword_count) << ")" << endl;
        tloss = 0;
        tword_count = 0;
      }
      if (++minibatch_count == minibatch_size) {
        sgd.update(1.0 / minibatch_size);
        minibatch_count = 0;
      }
      if (ctrlc_pressed) {
        break;
      }
    }
    if (ctrlc_pressed) {
      break;
    }
    auto dev_loss = ComputeLoss(*dev_bitext, attentional_model, t2s);
    cerr << "Iteration " << iteration + 1 << " loss: " << loss << " (perp=" << exp(loss/word_count) << ")" << " dev loss: " << dev_loss.first << endl;
    sgd.update_epoch();
    if (dev_loss.first <= best_dev_loss) {
      cerr << "New best!" << endl;
      Serialize(*train_bitext, attentional_model, model);
      best_dev_loss = dev_loss.first;
    }
  }

  return 0;
}

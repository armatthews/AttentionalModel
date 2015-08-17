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
  vector<vector<WordId> > source(bitext.size());
  vector<vector<WordId> > target(bitext.size());
  for (unsigned i = 0; i < bitext.size(); ++i) {
    source[i] = bitext.source_sentences[i];
    target[i] = bitext.target_sentences[i];
  }
  bitext.source_sentences = source;
  bitext.target_sentences = target;
}

void Serialize(Bitext& bitext, AttentionalModel& attentional_model, Model& model) {
  int r = ftruncate(fileno(stdout), 0);
  fseek(stdout, 0, SEEK_SET); 

  boost::archive::text_oarchive oa(cout);
  oa & bitext.source_vocab;
  oa & bitext.target_vocab;
  oa << attentional_model;
  oa << model;

}

pair<cnn::real, unsigned> ComputeLoss(Bitext& bitext, AttentionalModel& attentional_model) {
  cnn::real loss = 0.0;
  unsigned word_count = 0;
  for (unsigned i = 0; i < bitext.size(); ++i) {
    vector<WordId> source_sentence = bitext.source_sentences[i];
    vector<WordId> target_sentence = bitext.target_sentences[i];
    word_count += bitext.target_sentences[i].size() - 1; // Minus one for <s>
    ComputationGraph hg;
    attentional_model.BuildGraph(source_sentence, target_sentence, hg);
    double l = as_scalar(hg.forward());
    loss += l;
    if (ctrlc_pressed) {
      break;
    }
  }
  return make_pair(loss, word_count);
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
  ("train_bitext", po::value<string>()->required(), "Training bitext in source ||| target format")
  ("dev_bitext", po::value<string>()->default_value(""), "(Optional) Dev bitext, used for early stopping")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
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

  Bitext train_bitext;
  ReadCorpus(train_bitext_filename, train_bitext, true);
  cerr << "Read " << train_bitext.size() << " lines from " << train_bitext_filename << endl;
  cerr << "Vocab size: " << train_bitext.source_vocab.size() << "/" << train_bitext.target_vocab.size() << endl; 

  Bitext dev_bitext; 
  // TODO: The vocabulary objects really need to be tied. This is a really ghetto way of doing it
  dev_bitext.source_vocab = train_bitext.source_vocab;
  dev_bitext.target_vocab = train_bitext.target_vocab;
  unsigned initial_source_vocab_size = dev_bitext.source_vocab.size();
  unsigned initial_target_vocab_size = dev_bitext.target_vocab.size();
  ReadCorpus(dev_bitext_filename, dev_bitext, true);
  // Make sure the vocabulary sizes didn't change. If the dev set contains any words not in the training set, that's a problem!
  assert (initial_source_vocab_size == dev_bitext.source_vocab.size());
  assert (initial_target_vocab_size == dev_bitext.target_vocab.size());
  cerr << "Read " << dev_bitext.size() << " lines from " << dev_bitext_filename << endl;
  cerr << "Vocab size: " << dev_bitext.source_vocab.size() << "/" << dev_bitext.target_vocab.size() << endl;

  cnn::Initialize(argc, argv);
  std::mt19937 rndeng(42);
  Model model;
  AttentionalModel attentional_model(model, train_bitext.source_vocab.size(), train_bitext.target_vocab.size());
  //SimpleSGDTrainer sgd(&model, 0.0, 0.1);
  //AdagradTrainer sgd(&model, 0.0, 0.1);
  //AdadeltaTrainer sgd(&model, 0.0);
  //AdadeltaTrainer sgd(&model, 0.0, 1e-6, 0.992);
  //RmsPropTrainer sgd(&model, 0.0, 0.1);
  AdamTrainer sgd(&model, 1.e-4);
  sgd.eta_decay = 0.05;

  cerr << "Training model...\n";
  unsigned minibatch_count = 0;
  const unsigned minibatch_size = 1;
  cnn::real best_dev_loss = numeric_limits<cnn::real>::max();
  for (unsigned iteration = 0; iteration < num_iterations; iteration++) {
    unsigned word_count = 0;
    unsigned tword_count = 0;
    shuffle(train_bitext, rndeng);
    double loss = 0.0;
    double tloss = 0.0;
    for (unsigned i = 0; i < train_bitext.size(); ++i) {
      //cerr << "Reading sentence pair #" << i << endl;
      vector<WordId> source_sentence = train_bitext.source_sentences[i];
      vector<WordId> target_sentence = train_bitext.target_sentences[i];
      word_count += train_bitext.target_sentences[i].size() - 1; // Minus one for <s>
      tword_count += train_bitext.target_sentences[i].size() - 1; // Minus one for <s>
      ComputationGraph hg;
      attentional_model.BuildGraph(source_sentence, target_sentence, hg);
      double l = as_scalar(hg.forward());
      loss += l;
      tloss += l;
      hg.backward();
      if (i % 50 == 0 && i > 0) {
        float fractional_iteration = (float)iteration + ((float)i / train_bitext.size());
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
    auto dev_loss = ComputeLoss(dev_bitext, attentional_model);
    cerr << "Iteration " << iteration + 1 << " loss: " << loss << " (perp=" << exp(loss/word_count) << ")" << " dev loss: " << dev_loss.first << endl;
    sgd.update_epoch();
    if (dev_loss.first <= best_dev_loss) {
      cerr << "New best!" << endl;
      Serialize(train_bitext, attentional_model, model);
      best_dev_loss = dev_loss.first;
    }
  }

  //Serialize(train_bitext, attentional_model, model);
  return 0;
}

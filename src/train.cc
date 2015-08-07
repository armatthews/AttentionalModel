#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_oarchive.hpp>

#include <iostream>
#include <fstream>
#include <csignal>
#include <algorithm>

#include "bitext.h"
#include "attentional.h"

using namespace cnn;
using namespace std;

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
  ftruncate(fileno(stdout), 0);
  fseek(stdout, 0, SEEK_SET); 

  boost::archive::text_oarchive oa(cout);
  oa & bitext.source_vocab;
  oa & bitext.target_vocab;
  oa << attentional_model;
  oa << model;

}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " corpus.txt" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);

  const string corpus_filename = argv[1];
  Bitext bitext;
  ReadCorpus(corpus_filename, bitext, true);
  cerr << "Read " << bitext.size() << " lines from " << corpus_filename << endl;
  cerr << "Vocab size: " << bitext.source_vocab.size() << "/" << bitext.target_vocab.size() << endl; 

  cnn::Initialize(argc, argv);
  std::mt19937 rndeng(42);
  Model model;
  AttentionalModel attentional_model(model, bitext.source_vocab.size(), bitext.target_vocab.size());
  //SimpleSGDTrainer sgd(&model, 0.0, 0.1);
  //AdagradTrainer sgd(&model, 0.0, 0.1);
  AdadeltaTrainer sgd(&model, 0.0);
  //AdadeltaTrainer sgd(&model, 0.0, 1e-6, 0.992);
  //RmsPropTrainer sgd(&model, 0.0, 0.1);
  sgd.eta_decay = 0.05;

  cerr << "Training model...\n";
  unsigned minibatch_count = 0;
  const unsigned minibatch_size = 1;
  for (unsigned iteration = 0; iteration < 1000 || false; iteration++) {
    unsigned word_count = 0;
    unsigned tword_count = 0;
    shuffle(bitext, rndeng);
    double loss = 0.0;
    double tloss = 0.0;
    for (unsigned i = 0; i < bitext.size(); ++i) {
      //cerr << "Reading sentence pair #" << i << endl;
      vector<WordId> source_sentence = bitext.source_sentences[i];
      vector<WordId> target_sentence = bitext.target_sentences[i];
      word_count += bitext.target_sentences[i].size() - 1; // Minus one for <s>
      tword_count += bitext.target_sentences[i].size() - 1; // Minus one for <s>
      ComputationGraph hg;
      attentional_model.BuildGraph(source_sentence, target_sentence, hg);
      double l = as_scalar(hg.forward());
      loss += l;
      tloss += l;
      hg.backward();
      if (i % 50 == 0) {
        cerr << "--" << iteration << '.' << ((float)i / bitext.size()) << " loss: " << tloss << " (perp=" << exp(tloss/tword_count) << ")" << endl;
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
    cerr << "Iteration " << iteration << " loss: " << loss << " (perp=" << exp(loss/word_count) << ")" << endl;
    sgd.update_epoch();
    Serialize(bitext, attentional_model, model);
  }

  Serialize(bitext, attentional_model, model);
  return 0;
}

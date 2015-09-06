#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/mp.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <csignal>
#include <random>
#include <memory>
#include <algorithm>

#include "bitext.h"
#include "attentional.h"

using namespace cnn;
using namespace cnn::mp;
using namespace std;
namespace po = boost::program_options;

// This function lets us elegantly handle the user pressing ctrl-c.
// We set a global flag, which causes the training loops to clean up
// and break. In particular, this allows models to be saved to disk
// before actually exiting the program.
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

// Dump all information about the trained model that will be required
// to decode with this model on a new set. This includes (at least)
// the source and target dictioanries, the attentional_model's layer
// sizes, and the CNN model's parameters.
void Serialize(Bitext* bitext, AttentionalModel& attentional_model, Model& model) {
  int r = ftruncate(fileno(stdout), 0);
  fseek(stdout, 0, SEEK_SET);

  boost::archive::text_oarchive oa(cout);
  // TODO: Serialize the t2s flag as part of the model, so predict/align/etc
  // can avoid having it as an argument.
  oa & *bitext->source_vocab;
  oa & *bitext->target_vocab;
  oa & attentional_model;
  oa & model;
}

// Reads in a bitext from a file. If parent is non-null, tie the dictionaries
// of the newly read bitext with the parent's. E.g. a dev set's dictionaries
// should be tied to the training set's so that they never differ.
Bitext* ReadBitext(const string& filename, Bitext* parent, bool t2s) {
  Bitext* bitext;
  if (t2s) {
    bitext = new T2SBitext(parent);
  }
  else {
    bitext = new S2SBitext(parent);
  }
  bitext->ReadCorpus(filename);
  cerr << "Read " << bitext->size() << " lines from " << filename << endl;
  cerr << "Vocab size: " << bitext->source_vocab->size() << "/" << bitext->target_vocab->size() << endl;
  return bitext;
}

Bitext* ReadBitext(const string& filename, bool t2s) {
  return ReadBitext(filename, nullptr, t2s);
}

template<class D>
class Learner : public ILearner<D> {
public:
  explicit Learner(Bitext* bitext, AttentionalModel& attentional_model, Model& model) : bitext(bitext), attentional_model(attentional_model), model(model) {}
  ~Learner() {}
  cnn::real LearnFromDatum(const D& datum, bool learn) {
    ComputationGraph cg;
    attentional_model.BuildGraph(datum.first, datum.second, cg);
    double loss = as_scalar(cg.forward());
    if (learn) {
      cg.backward();
    }
    return loss;
  }

  void SaveModel() {
    Serialize(bitext, attentional_model, model);
  }
private:
  Bitext* bitext;
  AttentionalModel& attentional_model;
  Model& model;
};

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
  ("random_seed,r", po::value<unsigned>()->default_value(0), "Random seed. If this value is 0 a seed will be chosen randomly.")
  ("cores,j", po::value<unsigned>()->default_value(1), "Number of CPU cores to use for training")
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
  const unsigned random_seed = vm["random_seed"].as<unsigned>();
  const bool t2s = vm["t2s"].as<bool>();
  const unsigned num_children = vm["cores"].as<unsigned>();

  Bitext* train_bitext = ReadBitext(train_bitext_filename, t2s);
  unsigned src_vocab_size = train_bitext->source_vocab->size();
  unsigned tgt_vocab_size = train_bitext->target_vocab->size();
  Bitext* dev_bitext = ReadBitext(dev_bitext_filename, train_bitext, t2s);
  assert (train_bitext->source_vocab->size() == src_vocab_size);
  assert (train_bitext->target_vocab->size() == tgt_vocab_size);

  cnn::Initialize(argc, argv, random_seed);
  std::mt19937 rndeng(42);
  Model model;
  AttentionalModel attentional_model(model, train_bitext->source_vocab->size(), train_bitext->target_vocab->size());
  SimpleSGDTrainer sgd(&model, 1e-4, 0.25);
  //AdagradTrainer sgd(&model, 0.0, 1.0);
  //AdadeltaTrainer sgd(&model, 0.0, 1e-6, 0.999);
  //RmsPropTrainer sgd(&model, 0.0, 1.0, 1e-20, 0.95);
  //AdamTrainer sgd(&model, 0.0, 0.001, 0.01, 0.9999, 1e-20); 
  //sgd.eta_decay = 0.01;
  //sgd.eta_decay = 0.5;

  unsigned dev_frequency = 5000;
  unsigned report_frequency = 50;
  if (t2s) {
    Learner<T2SBitext::SentencePair> learner(train_bitext, attentional_model, model);
    const auto& train_data = dynamic_cast<T2SBitext*>(train_bitext)->data;
    const auto& dev_data = dynamic_cast<T2SBitext*>(dev_bitext)->data;
    RunMultiProcess<T2SBitext::SentencePair>(num_children, &learner, &sgd, train_data, dev_data, num_iterations, dev_frequency, report_frequency);
  }
  else {
    Learner<S2SBitext::SentencePair> learner(train_bitext, attentional_model, model);
    const auto& train_data = dynamic_cast<S2SBitext*>(train_bitext)->data;
    const auto& dev_data = dynamic_cast<S2SBitext*>(dev_bitext)->data;
    RunMultiProcess<S2SBitext::SentencePair>(num_children, &learner, &sgd, train_data, dev_data, num_iterations, dev_frequency, report_frequency);
  }

  return 0;
}

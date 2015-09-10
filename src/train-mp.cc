#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/mp.h"
#include "cnn/grad-check.h"

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
#include "train.h"

using namespace cnn;
using namespace cnn::mp;
using namespace std;
namespace po = boost::program_options;

template<class D>
class Learner : public ILearner<D> {
public:
  explicit Learner(Bitext* bitext, AttentionalModel& attentional_model, Model& model) : bitext(bitext), attentional_model(attentional_model), model(model) {}
  ~Learner() {}
  cnn::real LearnFromDatum(const D& datum, bool learn) {
    ComputationGraph cg;
    attentional_model.BuildGraph(datum.first, datum.second, cg);
    //cnn::CheckGrad(model, cg);
    double loss = as_scalar(cg.forward());
    if (learn) {
      cg.backward();
    }
    return loss;
  }

  void SaveModel() {
    cerr << "Saving model..." << endl;
    Serialize(bitext, attentional_model, model);
    cerr << "Done saving model." << endl;
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
  // Optimizer configuration
  ("sgd", "Use SGD for optimization")
  ("momentum", po::value<double>(), "Use SGD with this momentum value")
  ("adagrad", "Use Adagrad for optimization")
  ("adadelta", "Use Adadelta for optimization")
  ("rmsprop", "Use RMSProp for optimization")
  ("adam", "Use Adam for optimization")
  ("learning_rate", po::value<double>(), "Learning rate for optimizer (SGD, Adagrad, Adadelta, and RMSProp only)")
  ("alpha", po::value<double>(), "Alpha (Adam only)")
  ("beta1", po::value<double>(), "Beta1 (Adam only)")
  ("beta2", po::value<double>(), "Beta2 (Adam only)")
  ("rho", po::value<double>(), "Moving average decay parameter (RMSProp and Adadelta only)")
  ("epsilon", po::value<double>(), "Epsilon value for optimizer (Adagrad, Adadelta, RMSProp, and Adam only)")
  ("regularization", po::value<double>()->default_value(0.0), "L2 Regularization strength")
  ("eta_decay", po::value<double>()->default_value(0.05), "Learning rate decay rate (SGD only)")
  ("no_clipping", "Disable clipping of gradients")
  // End optimizer configuration
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

  cnn::Initialize(argc, argv, random_seed, true);
  std::mt19937 rndeng(42);
  Model model;
  AttentionalModel attentional_model(model, train_bitext->source_vocab->size(), train_bitext->target_vocab->size());
  Trainer* sgd = CreateTrainer(model, vm);
  //SimpleSGDTrainer sgd(&model, 1e-4, 0.25);
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
    RunMultiProcess<T2SBitext::SentencePair>(num_children, &learner, sgd, train_data, dev_data, num_iterations, dev_frequency, report_frequency);
  }
  else {
    Learner<S2SBitext::SentencePair> learner(train_bitext, attentional_model, model);
    const auto& train_data = dynamic_cast<S2SBitext*>(train_bitext)->data;
    const auto& dev_data = dynamic_cast<S2SBitext*>(dev_bitext)->data;
    RunMultiProcess<S2SBitext::SentencePair>(num_children, &learner, sgd, train_data, dev_data, num_iterations, dev_frequency, report_frequency);
  }

  return 0;
}

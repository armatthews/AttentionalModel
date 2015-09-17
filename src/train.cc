#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>
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
using namespace std;
namespace po = boost::program_options;

// Builds the computation graph (according to attentional_model) of the i'th
// sentence pair in bitext. Returns the length of the target sentence in words.
unsigned BuildGraph(unsigned i, const Bitext* bitext, AttentionalModel& attentional_model, bool t2s, ComputationGraph& cg) {
  if (t2s) {
    const T2SBitext* t2s_bitext = dynamic_cast<const T2SBitext*>(bitext);
    auto& data_pair = t2s_bitext->GetDatum(i);
    attentional_model.BuildGraph(data_pair.first, data_pair.second, cg);
    return data_pair.second.size();
  }
  else {
    const S2SBitext* s2s_bitext = dynamic_cast<const S2SBitext*>(bitext);
    auto& data_pair = s2s_bitext->GetDatum(i);
    attentional_model.BuildGraph(data_pair.first, data_pair.second, cg);
    return data_pair.second.size();
  }
}

// Compute the sum of the loss values (according to attentional_model) of
// all the sentences in bitext. Returns a tuple that represents the loss
// value and the word count, which is useful to compute perplexities.
pair<cnn::real, unsigned> ComputeLoss(const Bitext* bitext, AttentionalModel& attentional_model, bool t2s) {
  cnn::real loss = 0.0;
  unsigned word_count = 0;
  for (unsigned i = 0; i < bitext->size(); ++i) {
    ComputationGraph cg;
    word_count += BuildGraph(i, bitext, attentional_model, t2s, cg) - 1; // Minus one for <s>
    double l = as_scalar(cg.forward());
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
  ("train_bitext", po::value<string>()->required(), "Training bitext in source_tree ||| target format")
  ("dev_bitext", po::value<string>()->default_value(""), "(Optional) Dev bitext, used for early stopping")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("batch_size,b", po::value<unsigned>()->default_value(1), "Size of minibatches")
  ("random_seed,r", po::value<unsigned>()->default_value(0), "Random seed. If this value is 0 a seed will be chosen randomly.")
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
  const unsigned batch_size = vm["batch_size"].as<unsigned>();
  const bool t2s = vm["t2s"].as<bool>();

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
  Trainer* sgd = CreateTrainer(model, vm);

  cerr << "Training model...\n";
  unsigned minibatch_count = 0;
  unsigned dev_freq_count = 0;
  const unsigned minibatch_size = std::min(batch_size, train_bitext->size());
  const unsigned dev_frequency = std::min(5000U, train_bitext->size());
  const unsigned report_frequency = 50;
  cnn::real best_dev_loss = numeric_limits<cnn::real>::max();
  for (unsigned iteration = 0; iteration < num_iterations; iteration++) {
    unsigned word_count = 0;
    unsigned tword_count = 0;
    train_bitext->Shuffle(rndeng);
    double loss = 0.0;
    double tloss = 0.0;
    for (unsigned i = 0; i < train_bitext->size(); ++i) {
      // These braces cause cg to go out of scope before we ever try to call
      // ComputeLoss() on the dev set. Without them, ComputeLoss() tries to
      // create a second ComputationGraph, which makes CNN quite unhappy.
      {
        ComputationGraph cg; 
        unsigned sent_word_count = BuildGraph(i, train_bitext, attentional_model, t2s, cg) - 1; // Minus one for <s>
        word_count += sent_word_count;
        tword_count += sent_word_count; 
        double sent_loss = as_scalar(cg.forward());
        loss += sent_loss;
        tloss += sent_loss;
        cg.backward();
      }
      if (i % report_frequency == report_frequency - 1) {
        float fractional_iteration = (float)iteration + ((float)(i + 1) / train_bitext->size());
        cerr << "--" << fractional_iteration << "     perp=" << exp(tloss/tword_count) << endl;
        cerr.flush();
        tloss = 0;
        tword_count = 0;
      }
      if (++minibatch_count == minibatch_size) {
        sgd->update(1.0 / minibatch_size);
        minibatch_count = 0;
      }
      if (++dev_freq_count == dev_frequency) {
        float fractional_iteration = (float)iteration + ((float)(i + 1) / train_bitext->size());
        auto dev_loss = ComputeLoss(dev_bitext, attentional_model, t2s);
        cnn::real dev_perp = exp(dev_loss.first / dev_loss.second);
        bool new_best = dev_loss.first <= best_dev_loss;
        cerr << "**" << fractional_iteration << " dev perp: " << dev_perp << (new_best ? " (New best!)" : "") << endl;
        cerr.flush();
        if (new_best) {
          Serialize(train_bitext, attentional_model, model);
          best_dev_loss = dev_loss.first;
        }
        else {
          //sgd->update_epoch();
        }
        dev_freq_count = 0;
      }
      if (ctrlc_pressed) {
        break;
      }
    }
    //sgd->update_epoch();
    cerr << "##" << (float)(iteration + 1) << "     perp=" << exp(loss / word_count) << endl;
    if (ctrlc_pressed) {
      break;
    }
  }

  return 0;
}

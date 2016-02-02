#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/mp.h"
#include "cnn/grad-check.h"
#include "cnn/cfsm-builder.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <csignal>
#include <random>
#include <memory>
#include <algorithm>

#include "translator.h"
#include "train.h"

using namespace cnn;
using namespace cnn::mp;
using namespace std;
namespace po = boost::program_options;

class Learner : public ILearner<SentencePair, SufficientStats> {
public:
  explicit Learner(const vector<Dict*>& dicts, Bitext& bitext, Translator<Sentence>& translator, Model& model) : dicts(dicts), bitext(bitext), translator(translator), model(model) {}
  ~Learner() {}
  SufficientStats LearnFromDatum(const SentencePair& datum, bool learn) {
    ComputationGraph cg;
    const Sentence& source = get<0>(datum);
    const Sentence& target = get<1>(datum);
    translator.BuildGraph(source, target, cg);

    cnn::real loss = as_scalar(cg.forward());
    unsigned word_count = target.size() - 1; // Minus one because we don't predict <s>
    if (learn) {
      cg.backward();
    }
    return SufficientStats(loss, word_count, 1);
  }

  void SaveModel() {
    cerr << "Saving model..." << endl;
    Serialize(dicts, translator, model);
    cerr << "Done saving model." << endl;
  }
private:
  vector<Dict*> dicts;
  Bitext& bitext;
  Translator<Sentence>& translator;
  Model& model;
};

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " corpus.txt" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);
  cnn::Initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("train_bitext", po::value<string>()->required(), "Training bitext in source_tree ||| target format")
  ("dev_bitext", po::value<string>()->required(), "Dev bitext, used for early stopping")
  ("clusters,c", po::value<string>()->default_value(""), "Vocabulary clusters file")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("cores,j", po::value<unsigned>()->default_value(1), "Number of CPU cores to use for training")
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
  ("model", po::value<string>(), "Reload this model and continue learning")
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
  const string clusters_filename = vm["clusters"].as<string>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const unsigned num_children = vm["cores"].as<unsigned>();

  Dict source_vocab;
  Dict target_vocab;
  Model cnn_model;
  Translator<Sentence>* translator = nullptr;

  if (vm.count("model")) {
    translator = new Translator<Sentence>();
    string filename = vm["model"].as<string>();
    vector<Dict*> dicts = {&source_vocab, &target_vocab};
    Deserialize(filename, dicts, *translator, cnn_model);
    assert (source_vocab.is_frozen());
    assert (target_vocab.is_frozen());
  }

  source_vocab.Convert("UNK");
  target_vocab.Convert("UNK");
  Bitext* train_bitext = ReadBitext(train_bitext_filename, &source_vocab, &target_vocab);
  if (train_bitext == nullptr) {
    return 1;
  }
  if (!source_vocab.is_frozen()) {
    source_vocab.Freeze();
    target_vocab.Freeze();
    source_vocab.SetUnk("UNK");
    target_vocab.SetUnk("UNK");
  }

  Bitext* dev_bitext = ReadBitext(dev_bitext_filename, &source_vocab, &target_vocab);
  if (dev_bitext == nullptr) {
    return 1;
  }

  if (!vm.count("model")) {
    unsigned embedding_dim = 3;
    unsigned annotation_dim = 10;
    unsigned alignment_hidden_dim = 7;
    unsigned output_state_dim = 11;

    EncoderModel<Sentence>* encoder_model = new BidirectionalSentenceEncoder(cnn_model, source_vocab.size(), embedding_dim, annotation_dim);
    AttentionModel* attention_model = new StandardAttentionModel(cnn_model, annotation_dim, output_state_dim, alignment_hidden_dim);
    OutputModel* output_model = new SoftmaxOutputModel(cnn_model, embedding_dim, annotation_dim, output_state_dim, &target_vocab, clusters_filename);
    translator = new Translator<Sentence>(encoder_model, attention_model, output_model);
    cerr << "Vocabulary sizes: " << source_vocab.size() << " / " << target_vocab.size() << endl;
  }

  // if we didn't load a model, then initialize parameters here

  unsigned dev_frequency = 500;
  unsigned report_frequency = 50;
  Learner learner({&source_vocab, &target_vocab}, *train_bitext, *translator, cnn_model);
  Trainer* sgd = CreateTrainer(cnn_model, vm);
  if (num_children > 1) {
    RunMultiProcess<SentencePair>(num_children, &learner, sgd, *train_bitext, *dev_bitext, num_iterations, dev_frequency, report_frequency);
  }
  else {
    RunSingleProcess<SentencePair>(&learner, sgd, *train_bitext, *dev_bitext, num_iterations, dev_frequency, report_frequency);
  }

  return 0;
}

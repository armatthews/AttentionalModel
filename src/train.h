#include "cnn/mp.h"
using namespace cnn;
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
    cnn::mp::stop_requested = true;
  }
}

// Dump all information about the trained model that will be required
// to decode with this model on a new set. This includes (at least)
// the source and target dictionaries, the attentional_model's layer
// sizes, and the CNN model's parameters.
void Serialize(Bitext* bitext, AttentionalModel& attentional_model, Model& model) {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {
    //cerr << "WARNING: Unable to truncate stdout. Error " << errno << endl;
  }
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
  unsigned initial_source_vocab_size = bitext->source_vocab->size();
  unsigned initial_target_vocab_size = bitext->target_vocab->size();
  bool success = bitext->ReadCorpus(filename);
  if (!success) {
    cerr << "Unable to read corpus \"" << filename << "\"" << endl;
    return nullptr;
  }
  cerr << "Read " << bitext->size() << " lines from " << filename << endl;
  cerr << "Vocab size: " << bitext->source_vocab->size() << "/" << bitext->target_vocab->size() << endl;
  if (parent != nullptr) {
    assert (bitext->source_vocab->size() == initial_source_vocab_size);
    assert (bitext->target_vocab->size() == initial_target_vocab_size);
  }
  return bitext;
}

Bitext* ReadBitext(const string& filename, bool t2s) {
  return ReadBitext(filename, nullptr, t2s);
}

Trainer* CreateTrainer(Model& model, const po::variables_map& vm) {
  double regularization_strength = vm["regularization"].as<double>();
  double eta_decay = vm["eta_decay"].as<double>();
  bool clipping_enabled = (vm.count("no_clipping") == 0);
  unsigned learner_count = vm.count("sgd") + vm.count("momentum") + vm.count("adagrad") + vm.count("adadelta") + vm.count("rmsprop") + vm.count("adam");
  if (learner_count > 1) {
    cerr << "Invalid parameters: Please specify only one learner type.";
    exit(1);
  }

  Trainer* trainer = NULL;
  if (vm.count("momentum")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.01;
    double momentum = vm["momentum"].as<double>();
    trainer = new MomentumSGDTrainer(&model, regularization_strength, learning_rate, momentum);
  }
  else if (vm.count("adagrad")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-20;
    trainer = new AdagradTrainer(&model, regularization_strength, learning_rate, eps);
  }
  else if (vm.count("adadelta")) {
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-6;
    double rho = (vm.count("rho")) ? vm["rho"].as<double>() : 0.95;
    trainer = new AdadeltaTrainer(&model, regularization_strength, eps, rho);
  }
  else if (vm.count("rmsprop")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-20;
    double rho = (vm.count("rho")) ? vm["rho"].as<double>() : 0.95;
    trainer = new RmsPropTrainer(&model, regularization_strength, learning_rate, eps, rho);
  }
  else if (vm.count("adam")) {
    double alpha = (vm.count("alpha")) ? vm["alpha"].as<double>() : 0.001;
    double beta1 = (vm.count("beta1")) ? vm["beta1"].as<double>() : 0.9;
    double beta2 = (vm.count("beta2")) ? vm["beta2"].as<double>() : 0.999;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-8;
    trainer = new AdamTrainer(&model, regularization_strength, alpha, beta1, beta2, eps);
  }
  else { /* sgd */
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    trainer = new SimpleSGDTrainer(&model, regularization_strength, learning_rate);
  }
  assert (trainer != NULL);

  trainer->eta_decay = eta_decay;
  trainer->clipping_enabled = clipping_enabled;
  return trainer;
}


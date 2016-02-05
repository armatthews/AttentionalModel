#include "translator.h"
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
// the source and target dictionaries, the translator's layer
// sizes, and the CNN model's parameters.
template <class Input>
void Serialize(vector<Dict*> dicts, Translator<Input>& translator, Model& model) {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {}
  fseek(stdout, 0, SEEK_SET);

  boost::archive::text_oarchive oa(cout);
  for (Dict* dict : dicts) {
    oa & *dict;
  }
  oa & model;
  oa & translator;
}

template <class Input>
void Deserialize(const string& filename, vector<Dict*> dicts, Translator<Input>& translator, Model& model) {
  ifstream f(filename);
  boost::archive::text_iarchive oa(f);
  for (Dict* dict : dicts) {
    oa & *dict;
  }
  oa & model;
  oa & translator;
}

Bitext* ReadBitext(const string& filename, Dict* source_vocab, Dict* target_vocab) {
  ifstream f(filename);
  if (!f.is_open()) {
    return nullptr;
  }

  WordId sBOS = source_vocab->Convert("<s>");
  WordId sEOS = source_vocab->Convert("</s>");
  WordId tBOS = target_vocab->Convert("<s>");
  WordId tEOS = target_vocab->Convert("</s>");

  Bitext* bitext = new Bitext();
  for (string line; getline(f, line);) {
    vector<string> parts = tokenize(line, "|||");
    assert (parts.size() == 2);

    Sentence source = ReadSentence(strip(parts[0]), source_vocab);
    source.insert(source.begin(), sBOS);
    source.push_back(sEOS);

    Sentence target = ReadSentence(strip(parts[1]), target_vocab);
    target.insert(target.begin(), tBOS);
    target.push_back(tEOS);

    bitext->push_back(make_pair(source, target));
  }

  cerr << "Read " << bitext->size() << " lines from " << filename << endl;
  return bitext;
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

class SufficientStats {
public:
  cnn::real loss;
  unsigned word_count;
  unsigned sentence_count;

  SufficientStats() : loss(), word_count(), sentence_count() {}

  SufficientStats(cnn::real loss, unsigned word_count, unsigned sentence_count) : loss(loss), word_count(word_count), sentence_count(sentence_count) {}

  SufficientStats& operator+=(const SufficientStats& rhs) {
    loss += rhs.loss;
    word_count += rhs.word_count;
    sentence_count += rhs.sentence_count;
    return *this;
  }

  friend SufficientStats operator+(SufficientStats lhs, const SufficientStats& rhs) {
    lhs += rhs;
    return lhs;
  }

  bool operator<(const SufficientStats& rhs) {
    return loss < rhs.loss;
  }

  friend std::ostream& operator<< (std::ostream& stream, const SufficientStats& stats) {
    return stream << exp(stats.loss / stats.word_count);
  }
};

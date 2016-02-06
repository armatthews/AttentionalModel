#include "cnn/cnn.h"
#include "cnn/mp.h"
#include <boost/program_options.hpp>

#include <iostream>
#include <csignal>
#include "syntax_tree.h"
#include "translator.h"
#include "tree_encoder.h"
#include "utils.h"

using namespace cnn;
using namespace cnn::expr;
using namespace cnn::mp;
using namespace std;
namespace po = boost::program_options;

void Serialize(const vector<Dict*>& dicts, const Translator& translator, Model& cnn_model);

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

class Learner : public ILearner<SentencePair, SufficientStats> {
public:
  Learner(const vector<Dict*>& dicts, Translator& translator, Model& cnn_model) :
    dicts(dicts), translator(translator), cnn_model(cnn_model) {}
  ~Learner() {}
  SufficientStats LearnFromDatum(const SentencePair& datum, bool learn) {
    ComputationGraph cg;
    TranslatorInput* input = get<0>(datum);
    Sentence* output = get<1>(datum);
    translator.BuildGraph(input, *output, cg);
    cnn::real loss = as_scalar(cg.forward());
    if (learn) {
      cg.backward();
    }
    return SufficientStats(loss, output->size() - 1, 1);
  }

  void SaveModel() {
    Serialize(dicts, translator, cnn_model);
  }
private:
  const vector<Dict*>& dicts;
  Translator& translator;
  Model& cnn_model;
};

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

void AddTrainerOptions(po::options_description& desc) {
  desc.add_options()
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
  ("no_clipping", "Disable clipping of gradients");
}

Trainer* CreateTrainer(Model& cnn_model, const po::variables_map& vm) {
  double regularization_strength = vm["regularization"].as<double>();
  double eta_decay = vm["eta_decay"].as<double>();
  bool clipping_enabled = (vm.count("no_clipping") == 0);
  unsigned learner_count = vm.count("sgd") + vm.count("momentum") + vm.count("adagrad") + vm.count("adadelta") + vm.count("rmsprop") + vm.count("adam");
  if (learner_count > 1) {
    cerr << "Invalid parameters: Please specify only one learner type.";
    exit(1);
  }

  Trainer* trainer = nullptr;
  if (vm.count("momentum")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.01;
    double momentum = vm["momentum"].as<double>();
    trainer = new MomentumSGDTrainer(&cnn_model, regularization_strength, learning_rate, momentum);
  }
  else if (vm.count("adagrad")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-20;
    trainer = new AdagradTrainer(&cnn_model, regularization_strength, learning_rate, eps);
  }
  else if (vm.count("adadelta")) {
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-6;
    double rho = (vm.count("rho")) ? vm["rho"].as<double>() : 0.95;
    trainer = new AdadeltaTrainer(&cnn_model, regularization_strength, eps, rho);
  }
  else if (vm.count("rmsprop")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-20;
    double rho = (vm.count("rho")) ? vm["rho"].as<double>() : 0.95;
    trainer = new RmsPropTrainer(&cnn_model, regularization_strength, learning_rate, eps, rho);
  }
  else if (vm.count("adam")) {
    double alpha = (vm.count("alpha")) ? vm["alpha"].as<double>() : 0.001;
    double beta1 = (vm.count("beta1")) ? vm["beta1"].as<double>() : 0.9;
    double beta2 = (vm.count("beta2")) ? vm["beta2"].as<double>() : 0.999;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-8;
    trainer = new AdamTrainer(&cnn_model, regularization_strength, alpha, beta1, beta2, eps);
  }
  else { /* sgd */
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    trainer = new SimpleSGDTrainer(&cnn_model, regularization_strength, learning_rate);
  }
  assert (trainer != nullptr);

  trainer->eta_decay = eta_decay;
  trainer->clipping_enabled = clipping_enabled;
  return trainer;
}

Sentence* ReadSentence(const string& line, Dict& dict) {
  vector<string> words = tokenize(strip(line), " ");
  Sentence* r = new Sentence();
  for (const string& w : words) {
    r->push_back(dict.Convert(w));
  }
  return r;
}

Bitext* ReadBitext(const string& filename, Dict& source_vocab, Dict& target_vocab) {
  ifstream f(filename);
  if (!f.is_open()) {
    return nullptr;
  }

  WordId sBOS = source_vocab.Convert("<s>");
  WordId sEOS = source_vocab.Convert("</s>");
  WordId tBOS = target_vocab.Convert("<s>");
  WordId tEOS = target_vocab.Convert("</s>");

  Bitext* bitext = new Bitext();
  for (string line; getline(f, line);) {
    vector<string> parts = tokenize(line, "|||");
    assert (parts.size() == 2);

    Sentence* source = ReadSentence(strip(parts[0]), source_vocab);
    source->insert(source->begin(), sBOS);
    source->push_back(sEOS);

    Sentence* target = ReadSentence(strip(parts[1]), target_vocab);
    target->insert(target->begin(), tBOS);
    target->push_back(tEOS);

    bitext->push_back(make_pair(source, target));
  }

  cerr << "Read " << bitext->size() << " lines from " << filename << endl;
  return bitext;
}

Bitext* ReadT2SBitext(const string& filename, Dict& source_vocab, Dict& target_vocab, Dict& label_vocab) {
  ifstream f(filename);
  if (!f.is_open()) {
    return nullptr;
  }

  source_vocab.Convert("<s>");
  source_vocab.Convert("</s>");
  WordId tBOS = target_vocab.Convert("<s>");
  WordId tEOS = target_vocab.Convert("</s>");

  Bitext* bitext = new Bitext();
  for (string line; getline(f, line);) {
    vector<string> parts = tokenize(line, "|||");
    assert (parts.size() == 2);

    SyntaxTree* source = new SyntaxTree(strip(parts[0]), &source_vocab, &label_vocab);
    source->AssignNodeIds();

    Sentence* target = ReadSentence(strip(parts[1]), target_vocab);
    target->insert(target->begin(), tBOS);
    target->push_back(tEOS);

    bitext->push_back(make_pair(source, target));
  }

  cerr << "Read " << bitext->size() << " lines from " << filename << endl;
  return bitext;
}

void Serialize(const vector<Dict*>& dicts, const Translator& translator, Model& cnn_model) {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {}
  fseek(stdout, 0, SEEK_SET);

  boost::archive::text_oarchive oa(cout);
  oa & cnn_model;
  for (const Dict* dict : dicts) {
    oa & dict;
  }
  oa & translator;
}

void Deserialize(const string& filename, vector<Dict*>& dicts, Translator& translator, Model& cnn_model) {
  ifstream f(filename);
  boost::archive::text_iarchive ia(f);
  ia & cnn_model;
  for (Dict* dict : dicts) {
    ia & dict;
  }
  ia & translator;
  f.close();
}

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);
  cnn::Initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("train_bitext", po::value<string>()->required(), "Training bitext in source_tree ||| target format")
  ("dev_bitext", po::value<string>()->required(), "Dev bitext, used for early stopping")
  ("clusters,c", po::value<string>()->default_value(""), "Vocabulary clusters file")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("cores,j", po::value<unsigned>()->default_value(1), "Number of CPU cores to use for training")
  ("t2s", "Use tree-to-string translation")
  ("model", po::value<string>(), "Reload this model and continue learning");

  AddTrainerOptions(desc);

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

  const bool t2s = vm.count("t2s");
  const unsigned num_cores = vm["cores"].as<unsigned>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const string train_bitext_filename = vm["train_bitext"].as<string>();
  const string dev_bitext_filename = vm["dev_bitext"].as<string>();
  cerr << "dev_bitext is currently " << dev_bitext_filename << endl;

  vector<Dict*> dicts;
  Model cnn_model;
  Translator* translator;
  Trainer* trainer = nullptr;

  Dict* source_vocab = new Dict();
  Dict* target_vocab = new Dict();
  Dict* label_vocab = nullptr;
  dicts.push_back(source_vocab);
  dicts.push_back(target_vocab);
  if (t2s) {
    label_vocab = new Dict();
    dicts.push_back(label_vocab);
  }

  if (vm.count("model")) {
    translator = new Translator();
    string model_filename = vm["model"].as<string>();
    Deserialize(model_filename, dicts, *translator, cnn_model);
  }

  Bitext* train_bitext = nullptr;
  Bitext* dev_bitext = nullptr;
  if (!t2s) {
    train_bitext = ReadBitext(train_bitext_filename, *source_vocab, *target_vocab);
    dev_bitext = ReadBitext(dev_bitext_filename, *source_vocab, *target_vocab);
  }
  else {
    train_bitext = ReadT2SBitext(train_bitext_filename, *source_vocab, *target_vocab, *label_vocab);
    dev_bitext = ReadT2SBitext(dev_bitext_filename, *source_vocab, *target_vocab, *label_vocab);
  }

  if (!vm.count("model")) {
    unsigned embedding_dim = 64;
    unsigned annotation_dim = 64;
    unsigned alignment_hidden_dim = 64;
    unsigned output_state_dim = 64;

    Dict* source_vocab = dicts[0];
    Dict* target_vocab = dicts[1];
    const string clusters_filename = vm["clusters"].as<string>(); 

    EncoderModel* encoder_model = nullptr;
    if (!t2s) {
      encoder_model = new BidirectionalSentenceEncoder(cnn_model, source_vocab->size(), embedding_dim, annotation_dim);
    }
    else {
      encoder_model = new TreeEncoder(cnn_model, source_vocab->size(), label_vocab->size(), embedding_dim, annotation_dim);
    }
    AttentionModel* attention_model = new StandardAttentionModel(cnn_model, annotation_dim, output_state_dim, alignment_hidden_dim);
    OutputModel* output_model = new SoftmaxOutputModel(cnn_model, embedding_dim, annotation_dim, output_state_dim, target_vocab, clusters_filename);
    translator = new Translator(encoder_model, attention_model, output_model);
  }

  cerr << "Vocabulary sizes: " << source_vocab->size() << " / " << target_vocab->size() << endl;

  trainer = CreateTrainer(cnn_model, vm);
  Learner learner(dicts, *translator, cnn_model);
  unsigned dev_frequency = 10000;
  unsigned report_frequency = 100;
  if (num_cores > 1) {
    RunMultiProcess<SentencePair>(num_cores, &learner, trainer, *train_bitext, *dev_bitext, num_iterations, dev_frequency, report_frequency);
  }
  else {
    RunSingleProcess<SentencePair>(&learner, trainer, *train_bitext, *dev_bitext, num_iterations, dev_frequency, report_frequency);
  }

  return 0;
}

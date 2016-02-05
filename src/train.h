#include <boost/program_options.hpp>
#include "syntax_tree.h"
#include "translator.h"
#include "cnn/mp.h"
using namespace cnn;
using namespace cnn::expr;
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
    cnn::mp::stop_requested = true;
  }
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

template <class Input>
class TranslatorTrainer : public ILearner<pair<Input, Sentence>, SufficientStats> {
public:
  typedef pair<Input, Sentence> SentencePair;
  typedef vector<SentencePair> Bitext;

  TranslatorTrainer();
  TranslatorTrainer(const po::variables_map& vm);

  SufficientStats LearnFromDatum(const SentencePair& datum, bool learn) override;
  void SaveModel() override;

  bool ReadBitext(const string& filename, Bitext* bitext);
  void Train(unsigned num_cores, unsigned num_iterations);
 
  void Save();
  bool Load(const string& filename);

private:
  Model cnn_model;
  vector<Dict*> dicts;
  Trainer* trainer;
  Translator<Input>* translator;

  Bitext* train_bitext;
  Bitext* dev_bitext;

  void CreateTrainer(const po::variables_map& vm);
  EncoderModel<Input>* CreateEncoder();
  void CreateTranslator(const po::variables_map& vm);

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int);
};

template <class Input>
TranslatorTrainer<Input>::TranslatorTrainer() {}

template <class Input>
TranslatorTrainer<Input>::TranslatorTrainer(const po::variables_map& vm) {
  if (vm.count("model")) {
    translator = new Translator<Input>();
    string filename = vm["model"].as<string>();
    Load(filename);
  }
}

template<class Input>
SufficientStats TranslatorTrainer<Input>::LearnFromDatum(const SentencePair& datum, bool learn) {
    ComputationGraph cg;
    const Input& source = get<0>(datum);
    const Sentence& target = get<1>(datum);
    translator->BuildGraph(source, target, cg);

    cnn::real loss = as_scalar(cg.forward());
    unsigned word_count = target.size() - 1; // Minus one because we don't predict <s>
    if (learn) {
      cg.backward();
    }
    return SufficientStats(loss, word_count, 1);
}

template<class Input>
void TranslatorTrainer<Input>::SaveModel() {
  Save();
}

template <>
bool TranslatorTrainer<Sentence>::ReadBitext(const string& filename, Bitext* bitext) {
  ifstream f(filename);
  if (!f.is_open()) {
    return false;
  }

  Dict* source_vocab = dicts[0];
  Dict* target_vocab = dicts[1];

  WordId sBOS = source_vocab->Convert("<s>");
  WordId sEOS = source_vocab->Convert("</s>");
  WordId tBOS = target_vocab->Convert("<s>");
  WordId tEOS = target_vocab->Convert("</s>");

  unsigned line_count = 0;
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
    line_count++;
  }

  cerr << "Read " << line_count << " lines from " << filename << endl;
  return true;
}

template <>
bool TranslatorTrainer<SyntaxTree>::ReadBitext(const string& filename, Bitext* bitext) {
  ifstream f(filename);
  if (!f.is_open()) {
    return false;
  }

  Dict* source_vocab = dicts[0];
  Dict* target_vocab = dicts[1];
  Dict* label_vocab = dicts[2];

  source_vocab->Convert("<s>");
  source_vocab->Convert("</s>");
  WordId tBOS = target_vocab->Convert("<s>");
  WordId tEOS = target_vocab->Convert("</s>");

  unsigned line_count = 0;
  for (string line; getline(f, line);) {
    vector<string> parts = tokenize(line, "|||");
    assert (parts.size() == 2);

    SyntaxTree source(strip(parts[0]), source_vocab, label_vocab);
    source.AssignNodeIds();

    Sentence target = ReadSentence(strip(parts[1]), target_vocab);
    target.insert(target.begin(), tBOS);
    target.push_back(tEOS);

    bitext->push_back(make_pair(source, target));
    line_count++;
  }

  cerr << "Read " << line_count << " lines from " << filename << endl;
  return true;
}

template <class Input>
void TranslatorTrainer<Input>::Train(unsigned num_cores, unsigned num_iterations) {
  unsigned dev_frequency = 10000;
  unsigned report_frequency = 100;
  if (num_cores > 1) {
    RunMultiProcess<SentencePair>(num_cores, this, trainer, *train_bitext, *dev_bitext, num_iterations, dev_frequency, report_frequency);
  }
  else {
    RunSingleProcess<SentencePair>(this, trainer, *train_bitext, *dev_bitext, num_iterations, dev_frequency, report_frequency);
  }
}

template <class Input>
void TranslatorTrainer<Input>::Save() {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {}
  fseek(stdout, 0, SEEK_SET);

  boost::archive::text_oarchive oa(cout);
  oa & *this;
}

template <class Input>
bool TranslatorTrainer<Input>::Load(const string& filename) {
  ifstream f(filename);
  boost::archive::text_iarchive oa(f);
  oa & *this;
  return true;
}

template <class Input>
void TranslatorTrainer<Input>::CreateTrainer(const po::variables_map& vm) {
  double regularization_strength = vm["regularization"].as<double>();
  double eta_decay = vm["eta_decay"].as<double>();
  bool clipping_enabled = (vm.count("no_clipping") == 0);
  unsigned learner_count = vm.count("sgd") + vm.count("momentum") + vm.count("adagrad") + vm.count("adadelta") + vm.count("rmsprop") + vm.count("adam");
  if (learner_count > 1) {
    cerr << "Invalid parameters: Please specify only one learner type.";
    exit(1);
  }

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
  assert (trainer != NULL);

  trainer->eta_decay = eta_decay;
  trainer->clipping_enabled = clipping_enabled;
}

template <>
EncoderModel<Sentence>* TranslatorTrainer<Sentence>::CreateEncoder() {
  return nullptr;
}

template <>
EncoderModel<SyntaxTree>* TranslatorTrainer<SyntaxTree>::CreateEncoder() {
  //EncoderModel<SyntaxTree>* encoder_model = new TreeEncoder(cnn_model, source_vocab.size(), label_vocab.size(), embedding_dim, annotation_dim);
  //return encoder_model;
  return nullptr;
}

template <class Input>
void TranslatorTrainer<Input>::CreateTranslator(const po::variables_map& vm) {
    unsigned embedding_dim = 64;
    unsigned annotation_dim = 64;
    unsigned alignment_hidden_dim = 64;
    unsigned output_state_dim = 64;

    Dict* source_vocab = dicts[0];
    Dict* target_vocab = dicts[1];
    const string clusters_filename = vm["clusters_filename"].as<string>(); 

    EncoderModel<Input>* encoder_model = CreateEncoder();
    AttentionModel* attention_model = new StandardAttentionModel(cnn_model, annotation_dim, output_state_dim, alignment_hidden_dim);
    OutputModel* output_model = new SoftmaxOutputModel(cnn_model, embedding_dim, annotation_dim, output_state_dim, target_vocab, clusters_filename);
    translator = new Translator<SyntaxTree>(encoder_model, attention_model, output_model);

    cerr << "Vocabulary sizes: " << source_vocab->size() << " / " << target_vocab->size() << endl;
}

template <class Input>
template <class Archive>
void TranslatorTrainer<Input>::serialize(Archive& ar, const unsigned int) {
  ar & cnn_model;
  ar & dicts;
  //ar & trainer;
  ar & translator;
}


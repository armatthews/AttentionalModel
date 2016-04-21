#include "train.h"

using namespace cnn;
using namespace cnn::expr;
using namespace cnn::mp;
using namespace std;
namespace po = boost::program_options;

class Learner : public ILearner<SentencePair, SufficientStats> {
public:
  Learner(const vector<Dict*>& dicts, Translator& translator, Model& cnn_model, bool quiet) :
    dicts(dicts), translator(translator), cnn_model(cnn_model), quiet(quiet) {}
  ~Learner() {}
  SufficientStats LearnFromDatum(const SentencePair& datum, bool learn) {
    ComputationGraph cg;
    TranslatorInput* input = get<0>(datum);
    Sentence* output = get<1>(datum);
    translator.BuildGraph(input, *output, cg);
    cnn::real loss = as_scalar(cg.forward());
    //cerr << "Loss of " << loss << " on a thing of length " << output->size() - 1 << endl;
    if (learn) {
      cg.backward();
    }
    return SufficientStats(loss, output->size() - 1, 1);
  }

  void SaveModel() {
    if (!quiet) {
      Serialize(dicts, translator, cnn_model);
    }
  }
private:
  const vector<Dict*>& dicts;
  Translator& translator;
  Model& cnn_model;
  bool quiet;
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

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);
  cnn::Initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("train_bitext", po::value<string>()->required(), "Training bitext in source_tree ||| target format")
  ("dev_bitext", po::value<string>()->required(), "Dev bitext, used for early stopping")
  ("hidden_size,h", po::value<unsigned>()->default_value(64), "Size of hidden layers")
  ("clusters,c", po::value<string>()->default_value(""), "Vocabulary clusters file")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("cores,j", po::value<unsigned>()->default_value(1), "Number of CPU cores to use for training")
  ("sparsemax", "Use Sparsemax (rather than Softmax) for computing attention")
  ("t2s", "Use tree-to-string translation")
  ("diagonal_prior", "Use diagonal prior on attention")
  ("coverage_prior", "Use coverage prior on attention")
  ("markov_prior", "Use Markov prior on attention (similar to the HMM model)")
  ("markov_prior_window_size", po::value<unsigned>()->default_value(5), "Window size to use for the Markov prior. A value of 5 indicates five buckets: -2 or more, -1, 0, +1, +2 or more.")
  ("use_fertility", "Use fertility instead of assuming one source word ≈ one output word. Affects coverage prior and syntax prior.")
  ("syntax_prior", "Use source-side syntax prior on attention")
  ("quiet,q", "Don't output model at all (useful during debugging)")
  ("dropout_rate", po::value<float>(), "Dropout rate (should be >= 0.0 and < 1)")
  ("report_frequency,r", po::value<unsigned>()->default_value(100), "Show the training loss of every r examples")
  ("dev_frequency,d", po::value<unsigned>()->default_value(10000), "Run the dev set every d examples. Save the model if the score is a new best")
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
  const bool quiet = vm.count("quiet");
  const bool sparsemax = vm.count("sparsemax");
  const unsigned num_cores = vm["cores"].as<unsigned>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const string train_bitext_filename = vm["train_bitext"].as<string>();
  const string dev_bitext_filename = vm["dev_bitext"].as<string>();
  const unsigned hidden_size = vm["hidden_size"].as<unsigned>();

  bool use_fertility = vm.count("use_fertility"); // TODO: Currently unused

  vector<Dict*> dicts;
  Model cnn_model;
  Translator* translator = nullptr;
  Trainer* trainer = nullptr;

  if (vm.count("model")) {
    translator = new Translator();
    string model_filename = vm["model"].as<string>();
    Deserialize(model_filename, dicts, *translator, cnn_model);
    for (Dict* dict : dicts) {
      assert (dict->is_frozen());
    }
  }
  else {
    Dict* source_vocab = new Dict();
    Dict* target_vocab = new Dict();
    Dict* label_vocab = nullptr;
    dicts.push_back(source_vocab);
    dicts.push_back(target_vocab);
    if (t2s) {
      label_vocab = new Dict();
      dicts.push_back(label_vocab);
    }
  }

  for (Dict* dict : dicts) {
    dict->Convert("UNK");
    dict->Convert("<s>");
    dict->Convert("</s>");
  }

  Dict* source_vocab = dicts[0];
  Dict* target_vocab = dicts[1];
  Dict* label_vocab = t2s ? dicts[2] : nullptr;

  Bitext* train_bitext = nullptr;
  Bitext* dev_bitext = nullptr;
  if (t2s || vm.count("syntax_prior")) {
    train_bitext = ReadT2SBitext(train_bitext_filename, *source_vocab, *target_vocab, *label_vocab);
    dev_bitext = ReadT2SBitext(dev_bitext_filename, *source_vocab, *target_vocab, *label_vocab);
  }
  else {
    train_bitext = ReadBitext(train_bitext_filename, *source_vocab, *target_vocab);
    dev_bitext = ReadBitext(dev_bitext_filename, *source_vocab, *target_vocab);
  }

  if (!vm.count("model")) {
    unsigned embedding_dim = hidden_size;
    unsigned annotation_dim = hidden_size;
    unsigned alignment_hidden_dim = hidden_size;
    unsigned output_state_dim = hidden_size;
    unsigned final_hidden_size = hidden_size;

    Dict* source_vocab = dicts[0];
    Dict* target_vocab = dicts[1];
    const string clusters_filename = vm["clusters"].as<string>(); 

    EncoderModel* encoder_model = nullptr;
    if (!t2s) {
      //encoder_model = new TrivialEncoder(cnn_model, source_vocab->size(), embedding_dim, annotation_dim);
      encoder_model = new BidirectionalSentenceEncoder(cnn_model, source_vocab->size(), embedding_dim, annotation_dim);
    }
    else {
      encoder_model = new TreeEncoder(cnn_model, source_vocab->size(), label_vocab->size(), embedding_dim, annotation_dim);
    }

    AttentionModel* attention_model = nullptr;
    if (!sparsemax) {
      attention_model = new StandardAttentionModel(cnn_model, source_vocab->size(), annotation_dim, output_state_dim, alignment_hidden_dim);
    }
    else {
      attention_model = new SparsemaxAttentionModel(cnn_model, source_vocab->size(), annotation_dim, output_state_dim, alignment_hidden_dim);
    }
    // attention_model = new EncoderDecoderAttentionModel(cnn_model, annotation_dim, output_state_dim);

    OutputModel* output_model = nullptr;
    // output_model = new SoftmaxOutputModel(cnn_model, embedding_dim, annotation_dim, output_state_dim, target_vocab, clusters_filename);
    output_model = new MlpSoftmaxOutputModel(cnn_model, embedding_dim, annotation_dim, output_state_dim, final_hidden_size, target_vocab, clusters_filename);

    translator = new Translator(encoder_model, attention_model, output_model);

    for (Dict* dict : dicts) {
      dict->Freeze();
      dict->SetUnk("UNK");
    }

    if (vm.count("coverage_prior")) {
      attention_model->AddPrior(new CoveragePrior(cnn_model));
    }

    if (vm.count("diagonal_prior")) {
      attention_model->AddPrior(new DiagonalPrior(cnn_model));
    }

    if (vm.count("markov_prior")) {
      unsigned window_size = vm["markov_prior_window_size"].as<unsigned>();
      attention_model->AddPrior(new MarkovPrior(cnn_model, window_size));
    }

    if (vm.count("syntax_prior")) {
      attention_model->AddPrior(new SyntaxPrior(cnn_model));
    }
  }

  if (vm.count("dropout_rate")) {
    translator->SetDropout(vm["dropout_rate"].as<float>());
  }

  cerr << "Vocabulary sizes: " << source_vocab->size() << " / " << target_vocab->size() << endl;
  cerr << "Total parameters: " << cnn_model.parameter_count() << endl;

  trainer = CreateTrainer(cnn_model, vm);
  Learner learner(dicts, *translator, cnn_model, quiet);
  unsigned dev_frequency = vm["dev_frequency"].as<unsigned>();
  unsigned report_frequency = vm["report_frequency"].as<unsigned>();
  if (num_cores > 1) {
    RunMultiProcess<SentencePair>(num_cores, &learner, trainer, *train_bitext, *dev_bitext, num_iterations, dev_frequency, report_frequency);
  }
  else {
    RunSingleProcess<SentencePair>(&learner, trainer, *train_bitext, *dev_bitext, num_iterations, dev_frequency, report_frequency);
  }

  return 0;
}

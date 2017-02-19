#include "train.h"

using namespace dynet;
using namespace dynet::expr;
using namespace dynet::mp;
using namespace std;
namespace po = boost::program_options;

enum InputType {kStandard = 0, kSyntaxTree = 1, kMorphology = 2, kRNNG = 3};

istream& operator>>(istream& in, InputType& input_type)
{
  std::string token;
  in >> token;

  // Lowercase the token
  transform(token.begin(), token.end(), token.begin(), ::tolower);

  if (token == "standard") {
    input_type = kStandard;
  }
  else if (token == "syntax") {
    input_type = kSyntaxTree;
  }
  else if (token == "morph") {
    input_type = kMorphology;
  }
  else if (token == "rnng") {
    input_type = kRNNG;
  }
  else {
    //throw boost::program_options::validation_error("Invalid input type!");
    assert (false);
  }
  return in;
}

class Learner : public ILearner<SentencePair, SufficientStats> {
public:
  Learner(const InputReader* const input_reader, const OutputReader* const output_reader, Translator& translator, Model& dynet_model, const Trainer* const trainer, float dropout_rate, bool quiet) :
    input_reader(input_reader), output_reader(output_reader), translator(translator), dynet_model(dynet_model), trainer(trainer), dropout_rate(dropout_rate), quiet(quiet) {}
  ~Learner() {}
  SufficientStats LearnFromDatum(const SentencePair& datum, bool learn) {
    ComputationGraph cg;
    InputSentence* input = get<0>(datum);
    OutputSentence* output = get<1>(datum);

    translator.SetDropout(learn ? dropout_rate : 0.0f);
    Expression loss_expr = translator.BuildGraph(input, output, cg);
    dynet::real loss = as_scalar(loss_expr.value());

    if (learn) {
      cg.backward(loss_expr);
    }

    return SufficientStats(loss, output->size(), 1);
  }

  void SaveModel() {
    if (!quiet) {
      Serialize(input_reader, output_reader, translator, dynet_model, trainer);
    }
  }
private:
  const InputReader* const input_reader;
  const OutputReader* const output_reader;
  Translator& translator;
  Model& dynet_model;
  const Trainer* const trainer;
  float dropout_rate;
  bool quiet;
};

InputReader* CreateInputReader(const po::variables_map& vm) {
  InputType input_type = vm["source_type"].as<InputType>();
  switch (input_type) {
    case kStandard:
      return new StandardInputReader();
      break;
    case kSyntaxTree:
      return new SyntaxInputReader();
      break;
    case kMorphology:
      return new MorphologyInputReader();
      break;
    default:
      assert (false && "Reader for unknown input type requested");
  }
  return nullptr;
}

OutputReader* CreateOutputReader(const po::variables_map& vm) {
  InputType input_type = vm["target_type"].as<InputType>();
  switch (input_type) {
    case kStandard:
      return new StandardOutputReader(vm["vocab"].as<string>());
      break;
    case kMorphology:
      return new MorphologyOutputReader(vm["vocab"].as<string>(), vm["root_vocab"].as<string>());
      break;
    case kRNNG:
      return new RnngOutputReader();
    default:
      assert (false && "Reader for unknown output type requested");
  }
  return nullptr;
}

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
    dynet::mp::stop_requested = true;
  }
}

int main(int argc, char** argv) {
  cerr << "Invoked as:";
  for (int i = 0; i < argc; ++i) {
    cerr << " " << argv[i];
  }
  cerr << "\n";

  signal (SIGINT, ctrlc_handler);
  dynet::initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")

  ("train_source", po::value<string>()->required(), "Training set source")
  ("train_target", po::value<string>()->required(), "Training set target")
  ("dev_source", po::value<string>()->required(), "Dev set source")
  ("dev_target", po::value<string>()->required(), "Dev set target")
  ("source_type", po::value<InputType>()->default_value(kStandard), "Source input type. One of \"standard\", for standard linear sentences, \"syntax\" for syntax trees, \"morph\" for morphologically analyzed sentences, \"rnng\" for recurrent neural network grammars")
  ("target_type", po::value<InputType>()->default_value(kStandard), "Target input type. One of the same choices as above")

  ("vocab,v", po::value<string>()->default_value(""), "Target vocabulary file. If specified, anything outside this list will be UNK'd. If unspecified, nothing will be UNK'd on the training set.")
  ("root_vocab", po::value<string>()->default_value(""), "Target root vocabulary file. Only used with the morphological output type")
  ("clusters,c", po::value<string>()->default_value(""), "Target vocabulary clusters file")
  ("root_clusters", po::value<string>()->default_value(""), "Target root vocabulary clusters file")
  ("cores,j", po::value<unsigned>()->default_value(1), "Number of CPU cores to use for training")

  ("peepconcat", "Concatenate the raw word vectors to the output of the encoder")
  ("peepadd", "Add the raw word vectors to the output of the encoder")
  ("key_size", po::value<unsigned>(), "Number of annotation dimensions to use to compute attention. Default is to use the whole annotation vector.")
  ("sparsemax", "Use Sparsemax (rather than Softmax) for computing attention")
  ("no_encoder_rnn", "Use raw word vectors instead of bidirectional RNN to encode")
  ("no_final_mlp", "Do not use an MLP between the attentional context vector and final softmax")
  ("diagonal_prior", "Use diagonal prior on attention")
  ("coverage_prior", "Use coverage prior on attention")
  ("markov_prior", "Use Markov prior on attention (similar to the HMM model)")
  ("markov_prior_window_size", po::value<unsigned>()->default_value(5), "Window size to use for the Markov prior. A value of 5 indicates five buckets: -2 or more, -1, 0, +1, +2 or more.")
  //("use_fertility", "Use fertility instead of assuming one source word ≈ one output word. Affects coverage prior and syntax prior.")
  ("syntax_prior", "Use source-side syntax prior on attention")

  ("hidden_size,h", po::value<unsigned>()->default_value(64), "Size of hidden layers")
  ("dropout_rate", po::value<float>()->default_value(0.0f), "Dropout rate (should be >= 0.0 and < 1)")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("quiet,q", "Don't output model at all (useful during debugging)")
  ("batch_size,b", po::value<unsigned>()->default_value(1), "Batch size (has no effect when using > 1 core)")
  ("report_frequency,r", po::value<unsigned>()->default_value(100), "Show the training loss of every r examples")
  ("dev_frequency,d", po::value<unsigned>()->default_value(10000), "Run the dev set every d examples. Save the model if the score is a new best")
  ("model", po::value<string>(), "Reload this model and continue learning");

  AddTrainerOptions(desc);

  po::positional_options_description positional_options;
  positional_options.add("train_source", 1);
  positional_options.add("train_target", 1);
  positional_options.add("dev_source", 1);
  positional_options.add("dev_target", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  //bool use_fertility = vm.count("use_fertility"); // TODO: Currently unused

  Model dynet_model;
  InputReader* input_reader = nullptr;
  OutputReader* output_reader = nullptr;
  Translator* translator = nullptr;
  Trainer* trainer = nullptr;

  if (vm.count("model")) {
    string model_filename = vm["model"].as<string>();
    translator = new Translator();
    Deserialize(model_filename, input_reader, output_reader, *translator, dynet_model, trainer);
    if (vm.count("sgd") || vm.count("adagrad") || vm.count("adam") || vm.count("rmsprop") || vm.count("momentum")) {
      trainer = CreateTrainer(dynet_model, vm);
    }
  }
  else {
    input_reader = CreateInputReader(vm);
    output_reader = CreateOutputReader(vm);
  }

  const string train_source_filename = vm["train_source"].as<string>();
  const string train_target_filename = vm["train_target"].as<string>();
  Bitext train_bitext = ReadBitext(train_source_filename, train_target_filename, input_reader, output_reader);
  input_reader->Freeze();
  output_reader->Freeze();

  const string dev_source_filename = vm["dev_source"].as<string>();
  const string dev_target_filename = vm["dev_target"].as<string>();
  Bitext dev_bitext = ReadBitext(dev_source_filename, dev_target_filename, input_reader, output_reader);

  const InputType source_type = vm["source_type"].as<InputType>();
  const InputType target_type = vm["target_type"].as<InputType>();

  if (!vm.count("model")) {
    bool peep_concat = vm.count("peepconcat") > 0;
    bool peep_add = vm.count("peepadd") > 0;
    const unsigned hidden_size = vm["hidden_size"].as<unsigned>();
    unsigned embedding_dim = hidden_size;
    unsigned encoder_lstm_dim = hidden_size;
    unsigned annotation_dim = peep_concat ? encoder_lstm_dim + embedding_dim : encoder_lstm_dim;
    unsigned alignment_hidden_dim = hidden_size;
    unsigned output_state_dim = hidden_size;
    unsigned final_hidden_size = hidden_size;

    const string clusters_filename = vm["clusters"].as<string>();

    EncoderModel* encoder_model = nullptr;
    if (source_type == kStandard || source_type == kMorphology) {

      Embedder* embedder = nullptr;
      if (source_type == kStandard) {
        const unsigned vocab_size = dynamic_cast<const StandardInputReader*>(input_reader)->vocab.size();
        embedder = new StandardEmbedder(dynet_model, vocab_size, embedding_dim);
      }
      else {
        const MorphologyInputReader* reader = dynamic_cast<const MorphologyInputReader*>(input_reader);
        const unsigned word_vocab_size = reader->word_vocab.size();
        const unsigned root_vocab_size = reader->root_vocab.size();
        const unsigned affix_vocab_size = reader->affix_vocab.size();
        const unsigned char_vocab_size = reader->char_vocab.size();
        const unsigned affix_emb_dim = 64;
        const unsigned char_emb_dim = embedding_dim;
        const unsigned affix_lstm_dim = 32;
        const unsigned char_lstm_dim = embedding_dim;
        const bool use_words = true;
        const bool use_morphology = false;
        embedder = new MorphologyEmbedder(dynet_model, word_vocab_size, root_vocab_size, affix_vocab_size, char_vocab_size, embedding_dim, affix_emb_dim, char_emb_dim, affix_lstm_dim, char_lstm_dim, use_words, use_morphology);
      }

      if (vm.count("no_encoder_rnn")) {
        encoder_model = new TrivialEncoder(dynet_model, embedder, encoder_lstm_dim);
      }
      else {
        encoder_model = new BidirectionalEncoder(dynet_model, embedder, encoder_lstm_dim, peep_concat, peep_add);
      }
    }
    else if (source_type == kSyntaxTree) {
      const Dict& source_vocab = dynamic_cast<const SyntaxInputReader*>(input_reader)->terminal_vocab;
      const Dict& label_vocab = dynamic_cast<const SyntaxInputReader*>(input_reader)->nonterminal_vocab;
      encoder_model = new TreeEncoder(dynet_model, source_vocab.size(), label_vocab.size(), embedding_dim, encoder_lstm_dim);
    }
    else {
      assert (false && "Unknown input type");
    }

    AttentionModel* attention_model = nullptr;
    const unsigned key_size = vm.count("key_size") > 0 ? vm["key_size"].as<unsigned>() : annotation_dim;
    if (!vm.count("sparsemax")) {
      attention_model = new StandardAttentionModel(dynet_model, annotation_dim, output_state_dim, alignment_hidden_dim, key_size);
    }
    else {
      attention_model = new SparsemaxAttentionModel(dynet_model, annotation_dim, output_state_dim, alignment_hidden_dim, key_size);
    }
    // attention_model = new EncoderDecoderAttentionModel(dynet_model, annotation_dim, output_state_dim);

    OutputModel* output_model = nullptr;
    if (target_type == kStandard) {
      Dict& target_vocab = dynamic_cast<StandardOutputReader*>(output_reader)->vocab;
      if (vm.count("no_final_mlp")) {
        output_model = new SoftmaxOutputModel(dynet_model, embedding_dim, annotation_dim, output_state_dim, &target_vocab, clusters_filename);
      }
      else {
        output_model = new MlpSoftmaxOutputModel(dynet_model, embedding_dim, annotation_dim, output_state_dim, final_hidden_size, &target_vocab, clusters_filename);
      }
    }
    else if (target_type == kMorphology) {
      MorphologyOutputReader* reader = dynamic_cast<MorphologyOutputReader*>(output_reader);
      const unsigned word_vocab_size = reader->word_vocab.size();
      const unsigned root_vocab_size = reader->root_vocab.size();
      const unsigned affix_vocab_size = reader->affix_vocab.size();
      const unsigned char_vocab_size = reader->char_vocab.size();
      const unsigned word_emb_dim = embedding_dim;
      const unsigned root_emb_dim = embedding_dim;
      const unsigned affix_emb_dim = 64;
      const unsigned char_emb_dim = 32;
      const unsigned model_chooser_hidden_dim = hidden_size;
      const unsigned affix_init_hidden_dim = hidden_size;
      const unsigned char_init_hidden_dim = hidden_size;
      const unsigned state_dim = hidden_size;
      const unsigned affix_lstm_dim = embedding_dim;
      const unsigned char_lstm_dim = 32;
      const string word_clusters_file = vm["clusters"].as<string>();
      const string root_clusters_file = vm["root_clusters"].as<string>();
      output_model = new MorphologyOutputModel(dynet_model, reader->word_vocab, reader->root_vocab, affix_vocab_size, char_vocab_size, word_emb_dim, root_emb_dim, affix_emb_dim, char_emb_dim, model_chooser_hidden_dim, affix_init_hidden_dim, char_init_hidden_dim, state_dim, affix_lstm_dim, char_lstm_dim, annotation_dim, word_clusters_file, root_clusters_file);
    }
    else if (target_type == kRNNG) {
      unsigned hidden_dim = hidden_size;
      unsigned term_emb_dim = embedding_dim;
      unsigned nt_emb_dim = embedding_dim;
      unsigned action_emb_dim = embedding_dim;
      Dict& target_vocab = dynamic_cast<RnngOutputReader*>(output_reader)->vocab;
      output_model = new RnngOutputModel(dynet_model, term_emb_dim, nt_emb_dim, action_emb_dim, annotation_dim, hidden_dim, &target_vocab, clusters_filename);
    }
    else {
      assert (false && "Unknown output type");
    }

    translator = new Translator(encoder_model, attention_model, output_model);

    if (vm.count("coverage_prior")) {
      attention_model->AddPrior(new CoveragePrior(dynet_model));
    }

    if (vm.count("diagonal_prior")) {
      attention_model->AddPrior(new DiagonalPrior(dynet_model));
    }

    if (vm.count("markov_prior")) {
      unsigned window_size = vm["markov_prior_window_size"].as<unsigned>();
      attention_model->AddPrior(new MarkovPrior(dynet_model, window_size));
    }

    if (vm.count("syntax_prior")) {
      attention_model->AddPrior(new SyntaxPrior(dynet_model));
    }
    trainer = CreateTrainer(dynet_model, vm);
  }

  //cerr << "Vocabulary sizes: " << source_vocab->size() << " / " << target_vocab->size() << endl;
  cerr << "Total parameters: " << dynet_model.parameter_count() << endl;

  const float dropout_rate = vm["dropout_rate"].as<float>();
  const bool quiet = vm.count("quiet");
  Learner learner(input_reader, output_reader, *translator, dynet_model, trainer, dropout_rate, quiet);

  const unsigned num_cores = vm["cores"].as<unsigned>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const unsigned batch_size = vm["batch_size"].as<unsigned>();
  unsigned dev_frequency = vm["dev_frequency"].as<unsigned>();
  unsigned report_frequency = vm["report_frequency"].as<unsigned>();
  if (num_cores > 1) {
    run_multi_process<SentencePair>(num_cores, &learner, trainer, train_bitext, dev_bitext, num_iterations, dev_frequency, report_frequency);
  }
  else {
    run_single_process<SentencePair>(&learner, trainer, train_bitext, dev_bitext, num_iterations, dev_frequency, report_frequency, batch_size);
  }

  return 0;
}

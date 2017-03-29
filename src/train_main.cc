#include <chrono>
#include "train.h"
#include "train_wrapper.h"

using namespace dynet;
using namespace dynet::expr;
using namespace dynet::mp;
using namespace std;
namespace po = boost::program_options;

TrainingWrapper* wrapper = nullptr;

int main(int argc, char** argv) {
  cerr << "Invoked as:";
  for (int i = 0; i < argc; ++i) {
    cerr << " " << argv[i];
  }
  cerr << "\n";

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
  //("use_fertility", "Use fertility instead of assuming one source word ≈ one output word. Affects coverage prior and syntax prior.") // TODO: Currently unused
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

  if (!vm.count("model")) {
    EncoderModel* encoder_model = CreateEncoderModel(vm, dynet_model, input_reader);
    AttentionModel* attention_model = CreateAttentionModel(vm, dynet_model);
    OutputModel* output_model = CreateOutputModel(vm, dynet_model, output_reader);
    translator = new Translator(encoder_model, attention_model, output_model);
    trainer = CreateTrainer(dynet_model, vm);
  }

  //cerr << "Vocabulary sizes: " << source_vocab->size() << " / " << target_vocab->size() << endl;
  cerr << "Total parameters: " << dynet_model.parameter_count() << endl;

  const float dropout_rate = vm["dropout_rate"].as<float>();
  const bool quiet = vm.count("quiet");
  Learner learner(input_reader, output_reader, *translator, dynet_model, trainer, dropout_rate, quiet);

  wrapper = new TrainingWrapper(train_bitext, dev_bitext, trainer, &learner);
  signal (SIGINT, [](int) { cerr << "ctrl-c pressed. Stopping..." << endl; wrapper->Stop(); } );
  wrapper->Train(vm);

  return 0;
}

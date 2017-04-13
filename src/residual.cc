#include "train.h"

using namespace dynet;
using namespace dynet::expr;
using namespace dynet::mp;
using namespace std;
namespace po = boost::program_options;

class ResidualModel {
public:
  ResidualModel();
  ResidualModel(EncoderModel* encoder_model, OutputModel* output_model, unsigned encoder_dim, unsigned state_dim, unsigned hidden_dim, Model& model);
  void NewGraph(ComputationGraph& cg);
  Expression BuildGraph(const InputSentence* const source, const OutputSentence* const target, const vector<float>& residuals, ComputationGraph& cg);
  Expression Predict(const InputSentence* const source, const OutputSentence* const target_prefix, ComputationGraph& cg);
private:
  EncoderModel* encoder_model;
  OutputModel* output_model;
  MLP* mlp;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & encoder_model;
    ar & output_model;
    ar & mlp;
  }
};

ResidualModel::ResidualModel() : encoder_model(nullptr), output_model(nullptr), mlp(nullptr) {}

ResidualModel::ResidualModel(EncoderModel* encoder_model, OutputModel* output_model, unsigned encoder_dim, unsigned state_dim, unsigned hidden_dim, Model& model) :
    encoder_model(encoder_model), output_model(output_model) {
  const unsigned input_dim = encoder_dim + state_dim;
  mlp = new MLP(model, input_dim, hidden_dim, 1);
}

void ResidualModel::NewGraph(ComputationGraph& cg) {
  encoder_model->NewGraph(cg);
  output_model->NewGraph(cg);
  mlp->NewGraph(cg);
} 

Expression ResidualModel::BuildGraph(const InputSentence* const source, const OutputSentence* const target, const vector<float>& residuals, ComputationGraph& cg) {
  NewGraph(cg);
  assert (target->size() == residuals.size());
  unsigned N = (unsigned)target->size();
  Expression source_embedding = encoder_model->EncodeSentence(source);
  Expression state = output_model->GetState();
  vector<Expression> losses(N);
  for (unsigned i = 0; i < N; ++i) {
    const shared_ptr<Word> word = target->at(i);
    state = output_model->AddInput(word, source_embedding);
    Expression pred = mlp->Feed(concatenate({source_embedding, state}));
    Expression loss = square(pred - residuals[i]);
    //cerr << "Pred: " << as_scalar(pred.value()) << ", target: " << residuals[i] << endl;
    losses[i] = loss;
  }
  return sum(losses);
}

void SerializeResidual(const InputReader* const input_reader, const OutputReader* const output_reader, const ResidualModel& residual_model, Model& dynet_model, const Trainer* const trainer) {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {}
  fseek(stdout, 0, SEEK_SET);

  boost::archive::binary_oarchive oa(cout);
  oa & dynet_model;
  oa & input_reader;
  oa & output_reader;
  oa & residual_model;
  oa & trainer;
}

void DeserializeResidual(const string& filename, InputReader*& input_reader, OutputReader*& output_reader, ResidualModel& residual_model, Model& dynet_model, Trainer*& trainer) {
  ifstream f(filename);
  boost::archive::binary_iarchive ia(f);
  ia & dynet_model;
  ia & input_reader;
  ia & output_reader;
  ia & residual_model;
  ia & trainer;
  f.close();
}

class SufficientStats {
public:
  dynet::real loss;
  unsigned word_count;
  unsigned sentence_count;

  SufficientStats();
  SufficientStats(dynet::real loss, unsigned word_count, unsigned sentence_count);
  SufficientStats& operator+=(const SufficientStats& rhs);
  SufficientStats operator+(const SufficientStats& rhs);
  bool operator<(const SufficientStats& rhs);
};
std::ostream& operator<< (std::ostream& stream, const SufficientStats& stats);

SufficientStats::SufficientStats() : loss(), word_count(), sentence_count() {}

SufficientStats::SufficientStats(dynet::real loss, unsigned word_count, unsigned sentence_count) : loss(loss), word_count(word_count), sentence_count(sentence_count) {}

SufficientStats& SufficientStats::operator+=(const SufficientStats& rhs) {
  loss += rhs.loss;
  word_count += rhs.word_count;
  sentence_count += rhs.sentence_count;
  return *this;
}

SufficientStats SufficientStats::operator+(const SufficientStats& rhs) {
  SufficientStats result = *this;
  result += rhs;
  return result;
}

bool SufficientStats::operator<(const SufficientStats& rhs) {
  return loss < rhs.loss;
}

std::ostream& operator<< (std::ostream& stream, const SufficientStats& stats) {
  return stream << stats.loss / stats.word_count << " (" << stats.loss << " over " << stats.word_count << " words)";
}

class ResidualLearner : public ILearner<SentencePair, SufficientStats> {
public:
  ResidualLearner(const InputReader* const input_reader, const OutputReader* const output_reader, Translator& translator, ResidualModel& residual_model, Model& dynet_model, const Trainer* const trainer, float dropout_rate, bool quiet) :
    input_reader(input_reader), output_reader(output_reader), translator(translator), residual_model(residual_model), dynet_model(dynet_model), trainer(trainer), dropout_rate(dropout_rate), quiet(quiet) {}
  ~ResidualLearner() {}
  SufficientStats LearnFromDatum(const SentencePair& datum, bool learn) {
    InputSentence* input = get<0>(datum);
    OutputSentence* output = get<1>(datum);
    vector<float> residual_targets;

    {
      ComputationGraph cg;
      translator.SetDropout(0.0f);
      vector<Expression> translator_loss_exprs = translator.PerWordLosses(input, output, cg);
      //cerr << "Got base model loses." << endl;
      vector<float> translator_losses(translator_loss_exprs.size());
      residual_targets.resize(translator_losses.size());
      for (unsigned i = 0; i < translator_losses.size(); ++i) {
        translator_losses[i] = as_scalar(translator_loss_exprs[i].value());
      }
      for (unsigned i = 1; i < translator_losses.size(); ++i) {
        unsigned j = translator_losses.size() - 1 - i;
        residual_targets[j] = translator_losses[j + 1] + residual_targets[j + 1];
      }
    }

    {
      ComputationGraph cg;
      //cerr << "Computed targets." << endl;
      Expression loss_expr = residual_model.BuildGraph(input, output, residual_targets, cg);
      //cerr << "Built residual graph." << endl;
      dynet::real loss = as_scalar(loss_expr.value());

      if (learn) {
        cg.backward(loss_expr);
        //cerr << "Computed backwards." << endl;
      }

      //cerr << "Done with datum." << endl;
      return SufficientStats(loss, output->size(), 1);
    }
  }

  void SaveModel() {
    if (!quiet) {
      SerializeResidual(input_reader, output_reader, residual_model, dynet_model, trainer);
    }
  }
private:
  const InputReader* const input_reader;
  const OutputReader* const output_reader;
  Translator& translator;
  ResidualModel& residual_model;
  Model& dynet_model;
  const Trainer* const trainer;
  float dropout_rate;
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

  ("model", po::value<string>()->required(), "Learn residuals with respect to this model")
  ("train_source", po::value<string>()->required(), "Training set source")
  ("train_target", po::value<string>()->required(), "Training set target")
  ("dev_source", po::value<string>()->required(), "Dev set source")
  ("dev_target", po::value<string>()->required(), "Dev set target")
  ("cores,j", po::value<unsigned>()->default_value(1), "Number of CPU cores to use for training")

  ("hidden_size,h", po::value<unsigned>()->default_value(64), "Size of hidden layers")
  ("dropout_rate", po::value<float>()->default_value(0.0f), "Dropout rate (should be >= 0.0 and < 1)")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("quiet,q", "Don't output model at all (useful during debugging)")
  ("batch_size,b", po::value<unsigned>()->default_value(1), "Batch size (has no effect when using > 1 core)")
  ("report_frequency,r", po::value<unsigned>()->default_value(100), "Show the training loss of every r examples")
  ("dev_frequency,d", po::value<unsigned>()->default_value(10000), "Run the dev set every d examples. Save the model if the score is a new best");

  AddTrainerOptions(desc);

  po::positional_options_description positional_options;
  positional_options.add("model", 1);
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

  Model dynet_model, dynet_model2;
  InputReader* input_reader = nullptr;
  OutputReader* output_reader = nullptr;
  Translator* translator = nullptr;
  Trainer* trainer = nullptr, *trainer2 = nullptr;

  string model_filename = vm["model"].as<string>();
  translator = new Translator();
  Deserialize(model_filename, input_reader, output_reader, *translator, dynet_model, trainer);

  const string train_source_filename = vm["train_source"].as<string>();
  const string train_target_filename = vm["train_target"].as<string>();
  Bitext train_bitext = ReadBitext(train_source_filename, train_target_filename, input_reader, output_reader);
  input_reader->Freeze();
  output_reader->Freeze();

  const string dev_source_filename = vm["dev_source"].as<string>();
  const string dev_target_filename = vm["dev_target"].as<string>();
  Bitext dev_bitext = ReadBitext(dev_source_filename, dev_target_filename, input_reader, output_reader);

  const unsigned vocab_size = dynamic_cast<const StandardInputReader*>(input_reader)->vocab.size();
  const unsigned hidden_size = vm["hidden_size"].as<unsigned>();
  Dict target_vocab = dynamic_cast<const StandardOutputReader*>(output_reader)->vocab;
  Embedder* embedder = new StandardEmbedder(dynet_model2, vocab_size, hidden_size);
  EncoderModel* encoder_model = new BidirectionalEncoder(dynet_model2, embedder, hidden_size, false, true);
  OutputModel* output_model = new SoftmaxOutputModel(dynet_model2, hidden_size, hidden_size, hidden_size, &target_vocab, "");
  
  ResidualModel* residual_model = new ResidualModel(encoder_model, output_model, hidden_size, hidden_size, hidden_size, dynet_model2);
  trainer2 = CreateTrainer(dynet_model2, vm);

  //cerr << "Vocabulary sizes: " << source_vocab->size() << " / " << target_vocab->size() << endl;
  cerr << "Total parameters: " << dynet_model2.parameter_count() << " (plus " << dynet_model.parameter_count() << " from the base model)" << endl;

  const float dropout_rate = vm["dropout_rate"].as<float>();
  const bool quiet = vm.count("quiet");
  ResidualLearner learner(input_reader, output_reader, *translator, *residual_model, dynet_model2, trainer, dropout_rate, quiet);

  const unsigned num_cores = vm["cores"].as<unsigned>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const unsigned batch_size = vm["batch_size"].as<unsigned>();
  unsigned dev_frequency = vm["dev_frequency"].as<unsigned>();
  unsigned report_frequency = vm["report_frequency"].as<unsigned>();
  if (num_cores > 1) {
    run_multi_process<SentencePair>(num_cores, &learner, trainer2, train_bitext, dev_bitext, num_iterations, dev_frequency, report_frequency);
  }
  else {
    run_single_process<SentencePair>(&learner, trainer2, train_bitext, dev_bitext, num_iterations, dev_frequency, report_frequency, batch_size);
  }

  return 0;
}

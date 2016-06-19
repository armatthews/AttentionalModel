#include "train.h"
#include "rnng.h"

typedef vector<Action> ActionSequence;

using namespace cnn;
using namespace cnn::expr;
using namespace cnn::mp;
using namespace std;
namespace po = boost::program_options;

void SerializeRNNG(const vector<Dict*>& dicts, ParserBuilder& parser, Model& cnn_model) {
}

vector<ActionSequence> ReadOracleActions(const string& filename, Dict& term_vocab, Dict& nt_vocab) {
  vector<ActionSequence> dataset;
  ifstream f(filename);
  assert (f.is_open() && "Unable to open oracle action file");

  for (string line; getline(f, line);) {
    vector<string> action_strings = tokenize(line, " ");
    vector<Action> actions;
    for (string action_string : action_strings) {
      action_string = strip(action_string);
      size_t open_paren_loc = action_string.find("(");
      if (open_paren_loc != string::npos) {
        string action_type_str = action_string.substr(0, open_paren_loc);
        string subtype_str = action_string.substr(open_paren_loc + 1, action_string.length() - open_paren_loc - 2);
        if (action_type_str == "SHIFT") {
          WordId subtype = term_vocab.Convert(subtype_str);
          actions.push_back({Action::kShift, subtype});
        }
        else if (action_type_str == "NT") {
          WordId subtype = nt_vocab.Convert(subtype_str);
          actions.push_back({Action::kNT, subtype});
        }
        else {
          assert (false && "Invalid action type with subtype");
        }
      }
      else {
        assert (action_string == "REDUCE" && "Invalid action type without subtype");
        actions.push_back({Action::kReduce, 0});
      }
    }
    dataset.push_back(actions);
  }

  return dataset;
}

class Learner : public ILearner<ActionSequence, SufficientStats> {
public:
  Learner(const vector<Dict*>& dicts, ParserBuilder& parser, Model& cnn_model, bool quiet) :
    dicts(dicts), parser(parser), cnn_model(cnn_model), quiet(quiet) {}
  ~Learner() {}
  SufficientStats LearnFromDatum(const ActionSequence& datum, bool learn) {
    ComputationGraph cg;
    parser.NewGraph(cg);
    parser.BuildGraph(datum);
    cnn::real loss = as_scalar(cg.forward());
    if (learn) {
      cg.backward();
    }
    return SufficientStats(loss, datum.size(), 1);
  }

  void SaveModel() {
    if (!quiet) {
      SerializeRNNG(dicts, parser, cnn_model);
    }
  }
private:
  const vector<Dict*>& dicts;
  ParserBuilder& parser;
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
  ("train_trees", po::value<string>()->required(), "Training trees")
  ("dev_trees", po::value<string>()->required(), "Dev trees, used for early stopping")
  ("hidden_size,h", po::value<unsigned>()->default_value(64), "Size of hidden layers")
  ("clusters,c", po::value<string>()->default_value(""), "Vocabulary clusters file")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("cores,j", po::value<unsigned>()->default_value(1), "Number of CPU cores to use for training")
  ("quiet,q", "Don't output model at all (useful during debugging)")
  ("dropout_rate", po::value<float>(), "Dropout rate (should be >= 0.0 and < 1)")
  ("report_frequency,r", po::value<unsigned>()->default_value(100), "Show the training loss of every r examples")
  ("dev_frequency,d", po::value<unsigned>()->default_value(10000), "Run the dev set every d examples. Save the model if the score is a new best")
  ("model", po::value<string>(), "Reload this model and continue learning");

  AddTrainerOptions(desc);

  po::positional_options_description positional_options;
  positional_options.add("train_trees", 1);
  positional_options.add("dev_trees", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const bool quiet = vm.count("quiet");
  const unsigned num_cores = vm["cores"].as<unsigned>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const string train_filename = vm["train_trees"].as<string>();
  const string dev_filename = vm["dev_trees"].as<string>();
  const unsigned hidden_size = vm["hidden_size"].as<unsigned>();

  vector<Dict*> dicts;
  Model cnn_model;
  ParserBuilder* parser = nullptr;
  Trainer* trainer = nullptr;

  if (vm.count("model")) {
    parser = new ParserBuilder();
    string model_filename = vm["model"].as<string>();
    //DeserializeRNNG(model_filename, dicts, *parser, cnn_model); // XXX
    for (Dict* dict : dicts) {
      assert (dict->is_frozen());
    }
  }
  else {
    Dict* term_vocab = new Dict();
    Dict* nt_vocab = new Dict();
    dicts.push_back(term_vocab);
    dicts.push_back(nt_vocab);
  }

  for (Dict* dict : dicts) {
    dict->Convert("UNK");
    dict->Convert("<s>");
    dict->Convert("</s>");
  }

  Dict* term_vocab = dicts[0];
  Dict* nt_vocab = dicts[1];

  vector<ActionSequence> training_set = ReadOracleActions(train_filename, *term_vocab, *nt_vocab);
  vector<ActionSequence> dev_set = ReadOracleActions(dev_filename, *term_vocab, *nt_vocab);

  if (!vm.count("model")) {
    unsigned hidden_dim = vm["hidden_size"].as<unsigned>();
    unsigned term_emb_dim = hidden_dim;
    unsigned nt_emb_dim = hidden_dim;
    unsigned action_emb_dim = hidden_dim;

    const string clusters_filename = vm["clusters"].as<string>();
    SoftmaxBuilder* cfsm = nullptr;
    if (clusters_filename.length() > 0) {
      cfsm = new ClassFactoredSoftmaxBuilder(hidden_dim, clusters_filename, term_vocab, &cnn_model);
    }
    else {
      cfsm = new StandardSoftmaxBuilder(hidden_dim, term_vocab->size(), &cnn_model);
    }

    parser = new ParserBuilder(cnn_model, cfsm, term_vocab->size(), nt_vocab->size(), nt_vocab->size() + 2, hidden_dim, term_emb_dim, nt_emb_dim, action_emb_dim);

    for (Dict* dict : dicts) {
      dict->Freeze();
      dict->SetUnk("UNK");
    }
  }

  if (vm.count("dropout_rate")) {
    parser->SetDropout(vm["dropout_rate"].as<float>());
  }

  cerr << "Vocabulary sizes: " << nt_vocab->size() << " NTs / " << term_vocab->size() << " terminals" << endl;
  cerr << "Total parameters: " << cnn_model.parameter_count() << endl;

  trainer = CreateTrainer(cnn_model, vm);
  Learner learner(dicts, *parser, cnn_model, quiet);
  unsigned dev_frequency = vm["dev_frequency"].as<unsigned>();
  unsigned report_frequency = vm["report_frequency"].as<unsigned>();
  if (num_cores > 1) {
    RunMultiProcess<ActionSequence>(num_cores, &learner, trainer, training_set, dev_set, num_iterations, dev_frequency, report_frequency);
  }
  else {
    RunSingleProcess<ActionSequence>(&learner, trainer, training_set, dev_set, num_iterations, dev_frequency, report_frequency);
  }

  return 0;
}

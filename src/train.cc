#include "train.h"

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
  else if (token == "dependency") {
    input_type = kDependency;
  }
  else {
    //throw boost::program_options::validation_error("Invalid input type!");
    assert (false);
  }
  return in;
}

void AddTrainerOptions(po::options_description& desc) {
  desc.add_options()
  ("sgd", "Use SGD for optimization")
  ("momentum", "Use SGD wiith momentum")
  ("adagrad", "Use Adagrad for optimization")
  ("adadelta", "Use Adadelta for optimization")
  ("rmsprop", "Use RMSProp for optimization")
  ("adam", "Use Adam for optimization")
  ("learning_rate", po::value<double>(), "Learning rate for optimizer (SGD, Adagrad, Adadelta, and RMSProp only)")
  ("alpha", po::value<double>()->default_value(0.001), "Alpha (Adam only)")
  ("gamma", po::value<double>()->default_value(0.9), "Momentum strength (Momentum only)")
  ("beta1", po::value<double>()->default_value(0.9), "Beta1 (Adam only)")
  ("beta2", po::value<double>()->default_value(0.999), "Beta2 (Adam only)")
  ("rho", po::value<double>(), "Moving average decay parameter (RMSProp and Adadelta only)")
  ("epsilon", po::value<double>(), "Epsilon value for optimizer (Adagrad, Adadelta, RMSProp, and Adam only)")
  ("eta_decay", po::value<double>()->default_value(0.05), "Learning rate decay rate (SGD only)")
  ("no_clipping", "Disable clipping of gradients");
}

Trainer* CreateTrainer(Model& dynet_model, const po::variables_map& vm) {
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
    double gamma = vm["gamma"].as<double>();
    trainer = new MomentumSGDTrainer(dynet_model, learning_rate, gamma);
  }
  else if (vm.count("adagrad")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-20;
    trainer = new AdagradTrainer(dynet_model, learning_rate, eps);
  }
  else if (vm.count("adadelta")) {
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-6;
    double rho = (vm.count("rho")) ? vm["rho"].as<double>() : 0.95;
    trainer = new AdadeltaTrainer(dynet_model, eps, rho);
  }
  else if (vm.count("rmsprop")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-20;
    double rho = (vm.count("rho")) ? vm["rho"].as<double>() : 0.95;
    trainer = new RmsPropTrainer(dynet_model, learning_rate, eps, rho);
  }
  else if (vm.count("adam")) {
    double alpha = (vm.count("alpha")) ? vm["alpha"].as<double>() : 0.001;
    double beta1 = (vm.count("beta1")) ? vm["beta1"].as<double>() : 0.9;
    double beta2 = (vm.count("beta2")) ? vm["beta2"].as<double>() : 0.999;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-8;
    trainer = new AdamTrainer(dynet_model, alpha, beta1, beta2, eps);
  }
  else { /* sgd */
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    trainer = new SimpleSGDTrainer(dynet_model, learning_rate);
  }
  assert (trainer != nullptr);

  trainer->eta_decay = eta_decay;
  trainer->clipping_enabled = clipping_enabled;
  return trainer;
}

InputReader* CreateInputReader(const po::variables_map& vm) {
  InputType input_type = vm["source_type"].as<InputType>();
  switch (input_type) {
    case kStandard:
      return new StandardInputReader(true);
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
      return new StandardOutputReader(vm["vocab"].as<string>(), true);
      break;
    case kMorphology:
      return new MorphologyOutputReader(vm["vocab"].as<string>(), vm["root_vocab"].as<string>());
      break;
    case kRNNG:
      return new RnngOutputReader(vm["vocab"].as<string>());
    case kDependency:
      return new StandardOutputReader(vm["vocab"].as<string>(), false);
      //return new DependencyOutputReader(vm["vocab"].as<string>());
    default:
      assert (false && "Reader for unknown output type requested");
  }
  return nullptr;
}

unsigned ComputeAnnotationDim(const po::variables_map& vm) {
  const bool peep_concat = vm.count("peepconcat") > 0;
  const bool peep_add = vm.count("peepadd") > 0;
  const unsigned hidden_size = vm["hidden_size"].as<unsigned>();
  const unsigned embedding_dim = hidden_size;
  const unsigned encoder_lstm_dim = hidden_size;
  const unsigned annotation_dim = peep_concat ? encoder_lstm_dim + embedding_dim : encoder_lstm_dim;
  return annotation_dim;
}

EncoderModel* CreateEncoderModel(const po::variables_map& vm, Model& dynet_model, InputReader* input_reader) {
  EncoderModel* encoder_model = nullptr;
  const InputType source_type = vm["source_type"].as<InputType>();
  const unsigned hidden_size = vm["hidden_size"].as<unsigned>();
  const unsigned embedding_dim = hidden_size;
  const unsigned encoder_lstm_dim = hidden_size;

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
      const bool peep_concat = vm.count("peepconcat") > 0;
      const bool peep_add = vm.count("peepadd") > 0;
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
  return encoder_model;
}

void AddPriors(const po::variables_map& vm, AttentionModel* attention_model, Model& dynet_model) {
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
}

AttentionModel* CreateAttentionModel(const po::variables_map& vm, Model& dynet_model) {
  AttentionModel* attention_model = nullptr;
  const unsigned hidden_size = vm["hidden_size"].as<unsigned>();
  const unsigned annotation_dim = ComputeAnnotationDim(vm);
  const unsigned key_size = vm.count("key_size") > 0 ? vm["key_size"].as<unsigned>() : annotation_dim;
  const unsigned alignment_hidden_dim = hidden_size;
  const unsigned output_state_dim = hidden_size;

  if (!vm.count("sparsemax")) {
    attention_model = new StandardAttentionModel(dynet_model, annotation_dim, output_state_dim, alignment_hidden_dim, key_size);
  }
  else {
    attention_model = new SparsemaxAttentionModel(dynet_model, annotation_dim, output_state_dim, alignment_hidden_dim, key_size);
  }

  AddPriors(vm, attention_model, dynet_model);
  return attention_model;
}

OutputModel* CreateOutputModel(const po::variables_map& vm, Model& dynet_model, OutputReader* output_reader) {
  OutputModel* output_model = nullptr;
  const InputType target_type = vm["target_type"].as<InputType>();
  const unsigned hidden_size = vm["hidden_size"].as<unsigned>();
  const unsigned embedding_dim = hidden_size;
  const unsigned annotation_dim = ComputeAnnotationDim(vm);
  const unsigned output_state_dim = hidden_size;
  const unsigned final_hidden_size = hidden_size;
  const string clusters_filename = vm["clusters"].as<string>();

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
  else if (target_type == kDependency) {
    Dict& target_vocab = dynamic_cast<StandardOutputReader*>(output_reader)->vocab;
    //Dict& target_vocab = dynamic_cast<DependencyOutputReader*>(output_reader)->vocab;
    Embedder* embedder = new StandardEmbedder(dynet_model, target_vocab.size(), embedding_dim);
    output_model = new DependencyOutputModel(dynet_model, embedder, annotation_dim, output_state_dim, final_hidden_size, target_vocab);
  }
  else {
    assert (false && "Unknown output type");
  }
  return output_model;
}


#include <iostream>
#include <chrono>
#include "train_wrapper.h"

using namespace std;

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
  return stream << exp(stats.loss / stats.word_count) << " (" << stats.loss << " over " << stats.word_count << " words)";
}

Learner::Learner(const InputReader* const input_reader, const OutputReader* const output_reader, Translator& translator, Model& dynet_model, const Trainer* const trainer, float dropout_rate, bool quiet) :
  input_reader(input_reader), output_reader(output_reader), translator(translator), dynet_model(dynet_model), trainer(trainer), dropout_rate(dropout_rate), quiet(quiet) {}

Learner::~Learner() {}

SufficientStats Learner::LearnFromDatum(const SentencePair& datum, bool learn) {
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

void Learner::SaveModel() {
  if (!quiet) {
    Serialize(input_reader, output_reader, translator, dynet_model, trainer);
  }
}


TrainingWrapper::TrainingWrapper(const Bitext& train_bitext, const Bitext& dev_bitext, Trainer* trainer, Learner* learner) :
    train_bitext(train_bitext), dev_bitext(dev_bitext), trainer(trainer), learner(learner),
    epoch(0), data_processed(0), sents_since_dev(0), first_dev_run(true), stop(false) {}

void TrainingWrapper::Train(const po::variables_map& vm) {
  const unsigned num_cores = vm["cores"].as<unsigned>();
  const unsigned num_epochs = vm["num_iterations"].as<unsigned>();
  //const unsigned batch_size = vm["batch_size"].as<unsigned>(); // TODO: Currently unused
  const unsigned dev_frequency = vm["dev_frequency"].as<unsigned>();
  const unsigned report_frequency = vm["report_frequency"].as<unsigned>();

  for (epoch = 0; epoch < num_epochs && !stop; ++epoch) {
    vector<unsigned> train_order = GenerateOrder(train_bitext.size());
    InitializeEpoch();
    for (unsigned start = 0; start < train_bitext.size() && !stop; start += report_frequency) {
      unsigned end = std::min((unsigned)train_bitext.size(), start + report_frequency);
      vector<SentencePair> train_slice(train_bitext.begin() + start, train_bitext.begin() + end);

      time_point start_time = GetTime();
      SufficientStats stats = RunSlice(train_slice, num_cores, true);
      time_point end_time = GetTime();
      double seconds_elapsed = GetSeconds(start_time, end_time);

      data_processed = end;
      Report(epoch, end, stats, seconds_elapsed);

      epoch_stats += stats;
      trainer->update(1.0);

      sents_since_dev += train_slice.size();
      if (sents_since_dev > dev_frequency) {
        RunDevSet(num_cores);
        sents_since_dev = 0;
      }
    }
    if (!stop) {
      FinalizeEpoch();
    }
  }
}

void TrainingWrapper::Stop() {
  stop = true;
}

TrainingWrapper::time_point TrainingWrapper::GetTime() {
  return chrono::steady_clock::now();
}

double TrainingWrapper::GetSeconds(TrainingWrapper::time_point& start, TrainingWrapper::time_point& end) {
  double seconds = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000.0;
  return seconds;
}

vector<unsigned> TrainingWrapper::GenerateOrder(unsigned size) {
  vector<unsigned> order(size);
  iota(order.begin(), order.end(), 0);
  shuffle(order.begin(), order.end(), *rndeng); 
  return order;
}

double TrainingWrapper::ComputeFractionalEpoch() const {
  double fractional_epoch = epoch + 1.0 * data_processed / train_bitext.size();
  return fractional_epoch;
}

void TrainingWrapper::Report(unsigned epoch, unsigned progress, SufficientStats& stats, double seconds_elapsed) {
  cerr << ComputeFractionalEpoch() << "\t" << "loss = " << stats << " (" << seconds_elapsed << "s)" << endl;
}

bool TrainingWrapper::RunDevSet(unsigned num_cores) {
  SufficientStats dev_stats = RunSlice(dev_bitext, num_cores, false);
  bool new_best = (first_dev_run || dev_stats < best_dev_stats);
  cerr << ComputeFractionalEpoch() << "\t" << "dev loss = " << dev_stats;
  cerr << (new_best ? " (New best!)" : "") << endl;
  if (new_best) {
    learner->SaveModel();
    best_dev_stats = dev_stats;
    first_dev_run = false;
  }
  return new_best;
}

void TrainingWrapper::InitializeEpoch() {
  epoch_stats = SufficientStats();
}

void TrainingWrapper::FinalizeEpoch() {
  cerr << "Epoch " << epoch + 1 << " average: " << epoch_stats << endl;
  trainer->update_epoch();
}

SufficientStats TrainingWrapper::RunSlice(const vector<SentencePair>& slice, unsigned num_cores, bool learn) {
  return run_sp_minibatch_trainer(learner, learn ? trainer : nullptr, slice);
  //return run_mp_minibatch_trainer(num_cores, learner, learn ? trainer : nullptr, slice);
}


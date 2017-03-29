#include <iostream>
#include "train_wrapper.h"

using namespace std;

TrainingWrapper::TrainingWrapper(const Bitext& train_bitext, const Bitext& dev_bitext, Trainer* trainer, Learner* learner) :
    train_bitext(train_bitext), dev_bitext(dev_bitext), trainer(trainer), learner(learner),
    epoch(0), data_processed(0), sents_since_dev(0), stop(false) {}

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

      std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
      SufficientStats stats = RunSlice(train_slice, num_cores, true);
      std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
      double seconds_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;

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
    FinalizeEpoch();
  }
}

void TrainingWrapper::Stop() {
  stop = true;
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

void TrainingWrapper::RunDevSet(unsigned num_cores) {
  SufficientStats dev_stats = RunSlice(dev_bitext, num_cores, false);
  cerr << ComputeFractionalEpoch() << "\t" << "dev loss = " << dev_stats << " (New best?)" << endl;
}

void TrainingWrapper::InitializeEpoch() {
  epoch_stats = SufficientStats();
}

void TrainingWrapper::FinalizeEpoch() {
  cerr << "Epoch " << epoch + 1 << " average: " << epoch_stats << endl;
  trainer->update_epoch();
}

SufficientStats TrainingWrapper::RunSlice(const vector<SentencePair>& slice, unsigned num_cores, bool learn) {
  return run_mp_minibatch_trainer(num_cores, learner, learn ? trainer : nullptr, slice);
}


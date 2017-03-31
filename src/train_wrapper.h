#pragma once
#include <boost/program_options.hpp>
#include <chrono>
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "train.h"
using namespace dynet;
namespace po = boost::program_options;

class TrainingWrapper {
public:
  typedef std::chrono::steady_clock::time_point time_point;

  TrainingWrapper(const Bitext& train_bitext, const Bitext& dev_bitext, Trainer* trainer, Learner* learner);
  void Train(const po::variables_map& vm);
  void Stop();

  static time_point GetTime();
  static double GetSeconds(time_point& start, time_point& end);
  static vector<unsigned> GenerateOrder(unsigned size);
  double ComputeFractionalEpoch() const;

private:
  void Report(unsigned epoch, unsigned progress, SufficientStats& stats, double seconds_elapsed);
  void RunDevSet(unsigned num_cores);
  void InitializeEpoch();
  void FinalizeEpoch();
  SufficientStats RunSlice(const vector<SentencePair>& slice, unsigned num_cores, bool learn);

  const Bitext& train_bitext;
  const Bitext& dev_bitext;
  Trainer* trainer;
  Learner* learner;

  unsigned epoch;
  unsigned data_processed;
  unsigned sents_since_dev;
  SufficientStats epoch_stats;
  SufficientStats best_dev_stats;
  volatile bool stop;
};


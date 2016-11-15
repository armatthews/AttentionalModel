#pragma once
#include <deque>
#include <utility>
#include <memory>
#include "dynet/dict.h"
#include "utils.h"

using namespace dynet;
using namespace std;

template <typename T>
class KBestList {
public:
  unsigned max_size;

  KBestList() : max_size(0) {}
  explicit KBestList(unsigned max_size) : max_size(max_size) {}

  bool add(double score, T hyp) {
    if (size() == 0) {
      hypotheses.push_back(make_pair(score, hyp));
      return true;
    }

    // Check if this item is worse than EVERYTHING currently in the k-best list
    if (score < hypotheses.back().first) {
      if (size() == max_size) {
        return false;
      }
      else {
        hypotheses.push_back(make_pair(score, hyp));
        return true;
      }
    }

    // Now we know that there exists atleast one item in the list worse than the new item
    // So we traverse the list from the end, and find the correct position to insert the
    // new item into.
    auto it = hypotheses.begin();
    for (; it != hypotheses.end() && it->first > score; ++it) {}
    hypotheses.insert(it, make_pair(score, hyp));
    if (size() > max_size) {
      hypotheses.pop_back();
    }
    return true;
  }

  unsigned size() const {
    return hypotheses.size();
  }

  const deque<pair<double, T> >& hypothesis_list() const {
    return hypotheses;
  }
private:
  deque<pair<double, T>> hypotheses;
};

class OutputReader;
void OutputKBestList(unsigned sentence_number, KBestList<shared_ptr<OutputSentence>> kbest, OutputReader* output_reader);

#include <boost/algorithm/string/join.hpp>
#include <iostream>
#include "kbestlist.h"
using namespace std;

void OutputKBestList(unsigned sentence_number, KBestList<vector<WordId>> kbest, Dict& target_vocab) {
  for (auto& scored_hyp : kbest.hypothesis_list()) {
    double score = scored_hyp.first;
    vector<WordId> hyp = scored_hyp.second;
    vector<string> words(hyp.size());
    for (unsigned i = 0; i < hyp.size(); ++i) {
      words[i] = target_vocab.Convert(hyp[i]);
    }
    string translation = boost::algorithm::join(words, " ");
    cout << sentence_number << " ||| " << translation << " ||| " << score << endl;
  }
  cout.flush();
}

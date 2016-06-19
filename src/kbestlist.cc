#include <boost/algorithm/string/join.hpp>
#include <iostream>
#include "kbestlist.h"
using namespace std;

void OutputKBestList(unsigned sentence_number, KBestList<OutputSentence*> kbest, Dict& target_vocab) {
  for (auto& scored_hyp : kbest.hypothesis_list()) {
    double score = scored_hyp.first;
    const OutputSentence* const hyp = scored_hyp.second;
    vector<string> words(hyp->size());
    for (unsigned i = 0; i < hyp->size(); ++i) {
      const StandardWord* w = dynamic_cast<const StandardWord*>(hyp->at(i));
      words[i] = target_vocab.Convert(w->id);
    }
    string translation = boost::algorithm::join(words, " ");
    cout << sentence_number << " ||| " << translation << " ||| " << score << endl;
  }
  cout.flush();
}

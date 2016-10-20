#include <iostream>
#include <boost/algorithm/string/join.hpp>
#include "kbestlist.h"
#include "io.h"
using namespace std;

void OutputKBestList(unsigned sentence_number, KBestList<OutputSentence*> kbest, OutputReader* output_reader) {
  for (auto& scored_hyp : kbest.hypothesis_list()) {
    double score = scored_hyp.first;
    const OutputSentence* const hyp = scored_hyp.second;
    vector<string> words(hyp->size());
    for (unsigned i = 0; i < hyp->size(); ++i) {
      words[i] = output_reader->ToString(hyp->at(i));
    }
    string translation = boost::algorithm::join(words, " ");
    cout << sentence_number << " ||| " << translation << " ||| " << score << endl;
  }
  cout.flush();
}

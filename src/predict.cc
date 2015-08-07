#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <fstream>
#include <csignal>

#include "bitext.h"
#include "attentional.h"
#include "utils.h"

using namespace cnn;
using namespace std;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

void trim(vector<string>& tokens, bool removeEmpty) {
  for (unsigned i = 0; i < tokens.size(); ++i) {
    boost::algorithm::trim(tokens[i]);
    if (tokens[i].length() == 0 && removeEmpty) {
      tokens.erase(tokens.begin() + i);
      --i;
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: cat source.txt | " << argv[0] << " model" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);

  const string model_filename = argv[1];
  ifstream model_file(model_filename);
  if (!model_file.is_open()) {
    cerr << "ERROR: Unable to open " << model_filename << endl;
    exit(1);
  }
  boost::archive::text_iarchive ia(model_file);

  cnn::Initialize(argc, argv);
  Dict source_vocab;
  Dict target_vocab;
  ia & source_vocab;
  ia & target_vocab;
  source_vocab.Freeze();
  target_vocab.Freeze();

  Model model;
  AttentionalModel attentional_model(model, source_vocab.size(), target_vocab.size());

  ia & attentional_model;
  ia & model;

  WordId ksSOS = source_vocab.Convert("<s>");
  WordId ksEOS = source_vocab.Convert("</s>");
  WordId ktSOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");

  string line;
  for (; getline(cin, line);) {
    vector<string> parts = tokenize(line, "|||");
    trim(parts, false);

    vector<string> tokens = tokenize(parts[0], " ");
    trim(tokens, true);

    vector<WordId> source(tokens.size());
    for (unsigned i = 0; i < tokens.size(); ++i) {
      source[i] = source_vocab.Convert(tokens[i]);
    }
    source.insert(source.begin(), ksSOS);
    source.insert(source.end(), ksEOS);

    cerr << "Read source sentence: " << boost::algorithm::join(tokens, " ") << endl;
    if (parts.size() > 1) {
      vector<string> reference = tokenize(parts[1], " ");
      trim(reference, true);
      cerr << "  Read reference: " << boost::algorithm::join(reference, " ") << endl;
    }

    unsigned beam_size = 10;
    unsigned max_length = 20;
    unsigned kbest_size = 10; 
    KBestList<vector<WordId> > kbest = attentional_model.TranslateKBest(source, ktSOS, ktEOS, kbest_size, beam_size, max_length);
    for (auto& scored_hyp : kbest.hypothesis_list()) {
      double score = scored_hyp.first;
      vector<WordId> hyp = scored_hyp.second;
      vector<string> words(hyp.size());
      for (unsigned i = 0; i < hyp.size(); ++i) {
        words[i] = target_vocab.Convert(hyp[i]);
      }
      string translation = boost::algorithm::join(words, " ");
      cout << score << "\t" << translation << endl;
    }
    continue;

    map<string, int> translations;
    for (unsigned j = 0; j < 1000; ++j) {
      vector<WordId> target = attentional_model.SampleTranslation(source, ktSOS, ktEOS, 10);
      vector<string> words(target.size());
      for  (unsigned i = 0; i < target.size(); ++i) {
        words[i] = target_vocab.Convert(target[i]);
      }
      string translation = boost::algorithm::join(words, " ");
      translations[translation]++;

      if (ctrlc_pressed) {
        break;
      }
    }

    vector<pair<int, string> > translations2;
    for (auto it = translations.begin(); it != translations.end(); ++it) {
      translations2.push_back(make_pair(it->second, it->first));
    }

    auto comp = [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second);};
    sort(translations2.begin(), translations2.end(), comp);

    for (auto it = translations2.begin(); it != translations2.end(); ++it) {
      cout << it->first << "\t" << it->second << endl;
    }
  }

  return 0;
}

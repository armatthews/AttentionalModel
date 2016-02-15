#pragma once
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <vector>
#include "cnn/dict.h"
#include "syntax_tree.h"
#include "translator.h"
#include "utils.h"

using namespace std;
using namespace cnn;

Sentence* ReadSentence(const string& line, Dict& dict);
Bitext* ReadBitext(const string& filename, Dict& source_vocab, Dict& target_vocab);
Bitext* ReadT2SBitext(const string& filename, Dict& source_vocab, Dict& target_vocab, Dict& label_vocab);

void Serialize(const vector<Dict*>& dicts, const Translator& translator, Model& cnn_model);
void Deserialize(const string& filename, vector<Dict*>& dicts, Translator& translator, Model& cnn_model);

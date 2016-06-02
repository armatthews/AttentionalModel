#pragma once
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <vector>
#include <functional>
#include "cnn/dict.h"
#include "syntax_tree.h"
#include "translator.h"
#include "utils.h"

using namespace std;
using namespace cnn;

typedef function<vector<InputSentence*> ()> InputReader;
typedef function<vector<OutputSentence*> ()> OutputReader;
typedef vector<InputSentence*> Corpus;

Bitext ReadBitext(InputReader SourceReader, OutputReader TargetReader);

Corpus ReadStandardText(const string& filename, Dict& dict, bool add_bos_eos = true);
Corpus ReadSyntaxTrees(const string& filename, Dict& terminal_dict, Dict& label_dict);
Corpus ReadMorphologyText(const string& filename, Dict& word_dict, Dict& root_dict, Dict& affix_dict, Dict& char_dict);

LinearSentence* ReadSentence(const string& line, Dict& dict, bool add_bos_eos = true);
SyntaxTree* ReadSyntaxTree(const string& line, Dict& terminal_dict, Dict& label_dict);
LinearSentence* ReadMorphSentence(const string& line, Dict& word_dict, Dict& root_dict, Dict& affix_dict, Dict& char_dict);

Bitext* ReadBitext(const string& filename, Dict& source_vocab, Dict& target_vocab);
Bitext* ReadT2SBitext(const string& filename, Dict& source_vocab, Dict& target_vocab, Dict& label_vocab);

void Serialize(const vector<Dict*>& dicts, const Translator& translator, Model& cnn_model);
void Deserialize(const string& filename, vector<Dict*>& dicts, Translator& translator, Model& cnn_model);

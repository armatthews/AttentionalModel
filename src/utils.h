#pragma once
#include <vector>
#include <map>
#include <string>
#include <tuple>
/*#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>*/
#include "dynet/dict.h"
#include "dynet/expr.h"

using namespace std;
using namespace dynet;
using namespace dynet::expr;

typedef int WordId;

struct Word {
  virtual ~Word();
};

struct StandardWord : public Word {
  explicit StandardWord(WordId id);
  WordId id;
};

class Analysis {
public:
  WordId root;
  vector<WordId> affixes;
};

struct MorphoWord : public Word {
  WordId word;
  vector<Analysis> analyses;
  vector<WordId> chars;
};

class InputSentence {
public:
  virtual ~InputSentence();
  // Returns the number of nodes returned when we embed this sentence
  virtual unsigned NumNodes() const = 0;
};

typedef vector<Word*> OutputSentence;

class LinearSentence : public InputSentence, public OutputSentence {
public:
  unsigned NumNodes() const;
};

typedef pair<InputSentence*, OutputSentence*> SentencePair;
typedef vector<SentencePair> Bitext;

unsigned Sample(const vector<float>& dist);

unsigned int UTF8Len(unsigned char x);
unsigned int UTF8StringLen(const string& x);

vector<string> tokenize(string input, string delimiter, unsigned max_times);
vector<string> tokenize(string input, string delimiter);
vector<string> tokenize(string input, char delimiter);

string strip(const string& input);
vector<string> strip(const vector<string>& input, bool removeEmpty = false);

map<string, double> parse_feature_string(string input);

float logsumexp(const vector<float>& v);
vector<Expression> MakeLSTMInitialState(Expression c, unsigned lstm_dim, unsigned lstm_layer_count);


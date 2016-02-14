#pragma once
#include <vector>
#include <map>
#include <string>
#include <tuple>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "cnn/dict.h"

using namespace std;
using namespace cnn;

typedef int WordId;
class TranslatorInput {
public:
  virtual ~TranslatorInput();
};
class Sentence : public vector<WordId>, public TranslatorInput {};
typedef pair<TranslatorInput*, Sentence*> SentencePair;
typedef vector<SentencePair> Bitext;

inline unsigned int UTF8Len(unsigned char x);
inline unsigned int UTF8StringLen(const string& x);

vector<string> tokenize(string input, string delimiter, unsigned max_times);
vector<string> tokenize(string input, string delimiter);
vector<string> tokenize(string input, char delimiter);

string strip(const string& input);
vector<string> strip(const vector<string>& input, bool removeEmpty = false);

map<string, double> parse_feature_string(string input);

float logsumexp(const vector<float>& v);

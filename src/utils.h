#pragma once
#include <vector>
#include <map>
#include <string>
#include <tuple>
#include "cnn/dict.h"

using namespace std;
using namespace cnn;

typedef int WordId;

inline unsigned int UTF8Len(unsigned char x);
inline unsigned int UTF8StringLen(const string& x);

vector<string> tokenize(string input, string delimiter, unsigned max_times);
vector<string> tokenize(string input, string delimiter);
vector<string> tokenize(string input, char delimiter);

string strip(const string& input);
vector<string> strip(const vector<string>& input, bool removeEmpty = false);

map<string, double> parse_feature_string(string input);

float logsumexp(const vector<float>& v);

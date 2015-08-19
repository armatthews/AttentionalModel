#pragma once
#include <vector>
#include <map>
#include <string>

using namespace std;

typedef int WordId;

inline unsigned int UTF8Len(unsigned char x);
inline unsigned int UTF8StringLen(const string& x);

vector<string> tokenize(string input, string delimiter, int max_times);
vector<string> tokenize(string input, string delimiter);
vector<string> tokenize(string input, char delimiter);

string strip(string input);
vector<string> strip(vector<string> input);

map<string, double> parse_feature_string(string input);

float logsumexp(const vector<float>& v);

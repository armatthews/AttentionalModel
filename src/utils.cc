#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <cassert>
#include <cctype>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>

using namespace std;

// given the first character of a UTF8 block, find out how wide it is
// see http://en.wikipedia.org/wiki/UTF-8 for more info
inline unsigned int UTF8Len(unsigned char x) {
  if (x < 0x80) return 1;
  else if ((x >> 5) == 0x06) return 2;
  else if ((x >> 4) == 0x0e) return 3;
  else if ((x >> 3) == 0x1e) return 4;
  else if ((x >> 2) == 0x3e) return 5;
  else if ((x >> 1) == 0x7e) return 6;
  else return 0;
}

inline unsigned int UTF8StringLen(const string& x) {
  unsigned pos = 0;
  int len = 0;
  while(pos < x.size()) {
    ++len;
    pos += UTF8Len(x[pos]);
  }
  return len;
}

vector<string> tokenize(string input, string delimiter, int max_times) {
  vector<string> tokens;
  //tokens.reserve(max_times);
  size_t last = 0;
  size_t next = 0;
  while ((next = input.find(delimiter, last)) != string::npos && tokens.size() < max_times) {
    tokens.push_back(input.substr(last, next-last));
    last = next + delimiter.length();
  }
  tokens.push_back(input.substr(last));
  return tokens;
}

vector<string> tokenize(string input, string delimiter) {
  return tokenize(input, delimiter, input.length());
}

vector<string> tokenize(string input, char delimiter) {
  return tokenize(input, string(1, delimiter));
}

string strip(string input) {
  size_t start = 0;
  size_t end = input.length();

  for (; start < input.length(); ++start) {
    if (!isspace(input[start])) {
      break;
    }
  }
 
  for (; end > 0; --end) {
    if (!isspace(input[end - 1])) {
      break;
    }
  }

  if (end <= start) {
    return "";
  }

  return input.substr(start, end - start);
}

vector<string> strip(vector<string> input) {
  vector<string> output(input.size());
  for (unsigned i = 0; i < input.size(); ++i) {
    output[i] = strip(input[i]);
  }
  return output;
}

map<string, double> parse_feature_string(string input) {
  map<string, double> output;
  for (string piece : tokenize(input, " ")) {
    vector<string> kvp = tokenize(piece, "=", 1);
    if (kvp.size() != 2) {
      cerr << "Invalid feature name-value pair: \"" << piece << "\n";
      //cerr << "Invalid feature name-value pair: \"" << boost::algorithm::join(kvp, "=") << "\n";
      exit(1);
    }

    string name = kvp[0];
    double value = stod(kvp[1]);
    output[name] = value;
  }
  return output;
}

float logsumexp(const vector<float>& v) {
  assert (v.size() > 0);
  float m = v[0];
  for (unsigned i = 1; i < v.size(); ++i) {
    if (v[i] < m) {
      m = v[i];
    }
  }

  float sum = 0.0f;
  for (unsigned i = 0; i < v.size(); ++i) {
    sum += exp(v[i] - m);
  }
  return m + log(sum);
}


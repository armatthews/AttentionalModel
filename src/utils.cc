#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <cassert>
#include <cctype>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>
#include "utils.h"

using namespace std;

InputSentence::~InputSentence() {}

unsigned LinearSentence::NumNodes() const {
  return this->size();
}

Word::~Word() {}
StandardWord::StandardWord(WordId id) : id(id) {}

MultiFactorInputSentence::MultiFactorInputSentence() {}

MultiFactorInputSentence::MultiFactorInputSentence(const vector<InputSentence*>& factors) : factors(factors) {}

void MultiFactorInputSentence::AddFactor(InputSentence* factor) {
  factors.push_back(factor);
}

const InputSentence* const MultiFactorInputSentence::GetFactor(unsigned i) const {
  assert (factors.size() > i);
  return factors[i];
}

unsigned MultiFactorInputSentence::NumFactors() const {
  return factors.size();
}

unsigned MultiFactorInputSentence::NumNodes() const {
  assert (factors.size() > 0);
  return factors[0]->NumNodes();
}

// Samples an item from a multinomial distribution
// The values in dist should sum to one.
unsigned Sample(const vector<float>& dist) {
  double r = rand01();
  unsigned w = 0;
  for (; w < dist.size(); ++w) {
    r -= dist[w];
    if (r < 0.0) {
      break;
    }
  }

  if (w == dist.size()) {
    --w;
  }
  return w;
}

// given the first character of a UTF8 block, find out how wide it is
// see http://en.wikipedia.org/wiki/UTF-8 for more info
unsigned int UTF8Len(unsigned char x) {
  if (x < 0x80) return 1;
  else if ((x >> 5) == 0x06) return 2;
  else if ((x >> 4) == 0x0e) return 3;
  else if ((x >> 3) == 0x1e) return 4;
  else if ((x >> 2) == 0x3e) return 5;
  else if ((x >> 1) == 0x7e) return 6;
  else return 0;
}

unsigned int UTF8StringLen(const string& x) {
  unsigned pos = 0;
  int len = 0;
  while(pos < x.size()) {
    ++len;
    pos += UTF8Len(x[pos]);
  }
  return len;
}

vector<string> tokenize(string input, string delimiter, unsigned max_times) {
  vector<string> tokens;
  //tokens.reserve(max_times);
  size_t last = 0;
  size_t next = 0;
  while ((next = input.find(delimiter, last)) != string::npos && tokens.size() < max_times) {
    tokens.push_back(input.substr(last, next-last));
    last = next + delimiter.length();
  }
  if (last != input.length()) {
    tokens.push_back(input.substr(last));
  }
  return tokens;
}

vector<string> tokenize(string input, string delimiter) {
  return tokenize(input, delimiter, input.length());
}

vector<string> tokenize(string input, char delimiter) {
  return tokenize(input, string(1, delimiter));
}

string strip(const string& input) {
  string output = input;
  boost::algorithm::trim(output);
  return output;
}

vector<string> strip(const vector<string>& input, bool removeEmpty) {
  vector<string> output;
  for (unsigned i = 0; i < input.size(); ++i) {
    string s = strip(input[i]);
    if (s.length() > 0 || !removeEmpty) {
      output.push_back(s);
    }
  }
  return output;
}

map<string, double> parse_feature_string(string input) {
  map<string, double> output;
  for (string piece : tokenize(input, " ")) {
    vector<string> kvp = tokenize(piece, "=", 1);
    if (kvp.size() != 2) {
      cerr << "Invalid feature name-value pair: \"" << piece << "\n";
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

vector<Expression> MakeLSTMInitialState(Expression c, unsigned lstm_dim, unsigned lstm_layer_count) {
  vector<Expression> hinit(lstm_layer_count * 2);
  for (unsigned i = 0; i < lstm_layer_count; ++i) {
    hinit[i] = pickrange(c, i * lstm_dim, (i + 1) * lstm_dim);
    hinit[i + lstm_layer_count] = tanh(hinit[i]);
  }
  return hinit;
}

string vec2str(Expression expr) {
  ostringstream oss;
  bool first = true;
  for (float f : as_vector(expr.value())) {
    oss << (first ? "" : " ") << f;
    first = false;
  }
  return oss.str();
}

bool same_value(Expression e1, Expression e2) {
  vector<float> v1 = as_vector(e1.value());
  vector<float> v2 = as_vector(e2.value());
  if (v1.size() != v2.size()) {
    return false;
  }
  for (unsigned i = 0; i < v1.size(); ++i) {
    if (v1[i] != v2[i]) {
      return false;
    }
  }
  return true;
}

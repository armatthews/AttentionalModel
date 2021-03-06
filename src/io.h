#pragma once
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <vector>
#include <functional>
#include "dynet/dict.h"
#include "dynet/training.h"
#include "syntax_tree.h"
#include "translator.h"
#include "utils.h"

using namespace std;
using namespace dynet;

class InputReader {
public:
  virtual vector<InputSentence*> Read(const string& filename) = 0;
  virtual void Freeze() = 0;
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class OutputReader {
public:
  virtual vector<OutputSentence*> Read(const string& filename) = 0;
  virtual string ToString(const shared_ptr<const Word> word) = 0;
  virtual void Freeze() = 0;
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class StandardInputReader : public InputReader {
public:
  StandardInputReader();
  explicit StandardInputReader(bool add_bos_eos);
  vector<InputSentence*> Read(const string& filename);
  void Freeze();
  Dict vocab;
private:
  bool add_bos_eos;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<InputReader>(*this);
    ar & vocab;
    ar & add_bos_eos;
  }
};
BOOST_CLASS_EXPORT_KEY(StandardInputReader)

class SyntaxInputReader : public InputReader {
public:
  vector<InputSentence*> Read(const string& filename);
  void Freeze();
  Dict terminal_vocab;
  Dict nonterminal_vocab;
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<InputReader>(*this);
    ar & terminal_vocab;
    ar & nonterminal_vocab;
  }
};
BOOST_CLASS_EXPORT_KEY(SyntaxInputReader)

class MorphologyInputReader : public InputReader {
public:
  vector<InputSentence*> Read(const string& filename);
  void Freeze();
  Dict word_vocab;
  Dict root_vocab;
  Dict affix_vocab;
  Dict char_vocab;
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<InputReader>(*this);
    ar & word_vocab;
    ar & root_vocab;
    ar & affix_vocab;
    ar & char_vocab;
  }
};
BOOST_CLASS_EXPORT_KEY(MorphologyInputReader)

class StandardOutputReader : public OutputReader {
public:
  StandardOutputReader();
  explicit StandardOutputReader(const string& vocab_file, bool add_bos_eos);
  vector<OutputSentence*> Read(const string& filename);
  string ToString(const shared_ptr<const Word> word);
  void Freeze();
  Dict vocab;
private:
  bool add_bos_eos;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<OutputReader>(*this);
    ar & vocab;
    ar & add_bos_eos;
  }
};
BOOST_CLASS_EXPORT_KEY(StandardOutputReader)

class MorphologyOutputReader : public OutputReader {
public:
  MorphologyOutputReader();
  MorphologyOutputReader(const string& vocab_file, const string& morph_vocab_file);
  vector<OutputSentence*> Read(const string& filename);
  string ToString(const shared_ptr<const Word> word);
  void Freeze();

  Dict word_vocab;
  Dict root_vocab;
  Dict affix_vocab;
  Dict char_vocab;
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<OutputReader>(*this);
    ar & word_vocab;
    ar & root_vocab;
    ar & affix_vocab;
    ar & char_vocab;
  }
};
BOOST_CLASS_EXPORT_KEY(MorphologyOutputReader)

class RnngOutputReader : public OutputReader {
public:
  RnngOutputReader();
  explicit RnngOutputReader(const string& vocab_file);
  vector<OutputSentence*> Read(const string& filename);
  string ToString(const shared_ptr<const Word> word);
  void Freeze();

  Dict vocab;
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<OutputReader>(*this);
    ar & vocab;
  }
};
BOOST_CLASS_EXPORT_KEY(RnngOutputReader)

class DependencyOutputReader : public OutputReader {
public:
  DependencyOutputReader();
  explicit DependencyOutputReader(const string& vocab_file);
  vector<OutputSentence*> Read(const string& filename);
  string ToString(const shared_ptr<const Word> word);
  void Freeze();

  Dict vocab;
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<OutputReader>(*this);
    ar & vocab;
  }
};
BOOST_CLASS_EXPORT_KEY(DependencyOutputReader)

void ReadDict(const string& filename, Dict& dict);
void ReadDictRnng(const string& filename, Dict& dict);
Bitext ReadBitext(const string& source_filename, const string& target_filename, InputReader* SourceReader, OutputReader* TargetReader);

void Serialize(const InputReader* const input_reader, const OutputReader* const output_reader, const Translator& translator, Model& dynet_model, const Trainer* const trainer);
void Deserialize(const string& filename, InputReader*& input_reader, OutputReader*& output_reader, Translator& translator, Model& dynet_model, Trainer*& trainer);

#include <fstream>
#include "io.h"
BOOST_CLASS_EXPORT_IMPLEMENT(StandardInputReader)
BOOST_CLASS_EXPORT_IMPLEMENT(SyntaxInputReader)
BOOST_CLASS_EXPORT_IMPLEMENT(MorphologyInputReader)
BOOST_CLASS_EXPORT_IMPLEMENT(StandardOutputReader)
BOOST_CLASS_EXPORT_IMPLEMENT(MorphologyOutputReader)
BOOST_CLASS_EXPORT_IMPLEMENT(RnngOutputReader)

LinearSentence* ReadStandardSentence(const string& line, Dict& dict, bool add_bos_eos) {
  vector<string> words = tokenize(strip(line), " ");
  LinearSentence* r = new LinearSentence();
  if (add_bos_eos) {
    r->push_back(new StandardWord(dict.Convert("<s>")));
  }
  for (const string& w : words) {
    r->push_back(new StandardWord(dict.Convert(w)));
  }
  if (add_bos_eos) {
    r->push_back(new StandardWord(dict.Convert("</s>")));
  }
  return r;
}

vector<LinearSentence*> ReadStandardSentences(const string& filename, Dict& dict, bool add_bos_eos) {
  ifstream f(filename);
  assert (f.is_open());

  vector<LinearSentence*> sentences;
  for (string line; getline(f, line);) {
    LinearSentence* sentence = ReadStandardSentence(strip(line), dict, add_bos_eos);
    sentences.push_back(sentence);
  }
  return sentences;
}

MorphoWord* ReadMorphoWord(const string& line, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab) {
  MorphoWord* word = new MorphoWord();
  vector<string> parts = tokenize(strip(line), "\t");
  string& word_str = parts[0];
  word->word = word_vocab.Convert(word_str);

  for (unsigned i = 1; i < parts.size(); ++i) {
    vector<string> morphemes = tokenize(parts[i], "+");
    assert (morphemes.size() > 0);
    Analysis analysis;
    analysis.root = root_vocab.Convert(morphemes[0]);
    for (unsigned j = 1; j < morphemes.size(); ++j) {
      analysis.affixes.push_back(affix_vocab.Convert(morphemes[j]));
    }
    word->analyses.push_back(analysis);
  }

  for (unsigned i = 0; i < word_str.length(); ) {
    unsigned len = UTF8Len(parts[0][i]);
    string c = parts[0].substr(i, len);
    word->chars.push_back(char_vocab.Convert(c));
    i += len;
  }
  return word;
}

LinearSentence* ReadMorphologySentence(const vector<string>& lines, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab, bool add_bos_eos) {
  LinearSentence* r = new LinearSentence();
  if (add_bos_eos) {
    // TODO: Add <s>
  }

  for (const string& line : lines) {
    r->push_back(ReadMorphoWord(line, word_vocab, root_vocab, affix_vocab, char_vocab));
  }

  if (add_bos_eos) {
    // TODO: Add </s>
  }
  return r;
}

vector<LinearSentence*> ReadMorphologySentences(const string& filename, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab, bool add_bos_eos) {
  ifstream f(filename);
  assert (f.is_open());

  vector<string> current_sentence;
  vector<LinearSentence*> sentences;
  for (string line; getline(f, line);) {
    string sline = strip(line);
    if (sline.length() == 0) {
      LinearSentence* sentence = ReadMorphologySentence(current_sentence, word_vocab, root_vocab, affix_vocab, char_vocab, add_bos_eos);
      sentences.push_back(sentence);
      current_sentence.clear();
    }
    else {
      current_sentence.push_back(sline);
    }
  }

  if (current_sentence.size() > 0) {
    LinearSentence* sentence = ReadMorphologySentence(current_sentence, word_vocab, root_vocab, affix_vocab, char_vocab, add_bos_eos);
    sentences.push_back(sentence);
    current_sentence.clear();
  }
  return sentences;
}

vector<InputSentence*> StandardInputReader::Read(const string& filename) {
  vector<LinearSentence*> corpus = ReadStandardSentences(filename, vocab, true);
  return vector<InputSentence*>(corpus.begin(), corpus.end());
}

vector<InputSentence*> SyntaxInputReader::Read(const string& filename) {
  ifstream f(filename);
  assert (f.is_open());

  vector<InputSentence*> sentences;
  for (string line; getline(f, line);) {
    SyntaxTree* sentence = new SyntaxTree(strip(line), &terminal_vocab, &nonterminal_vocab);
    sentence->AssignNodeIds();
    sentences.push_back(sentence);
  }

  return sentences;
}

vector<InputSentence*> MorphologyInputReader::Read(const string& filename) {
  vector<LinearSentence*> corpus = ReadMorphologySentences(filename, word_vocab, root_vocab, affix_vocab, char_vocab, true);
  return vector<InputSentence*>(corpus.begin(), corpus.end());
}

StandardOutputReader::StandardOutputReader() {}
StandardOutputReader::StandardOutputReader(const string& vocab_file) {
  vocab.Convert("UNK");
  vocab.Convert("<s>");
  vocab.Convert("</s>");
  if (vocab_file.length() > 0) {
    ReadDict(vocab_file, vocab);
    vocab.Freeze();
    vocab.SetUnk("UNK");
  }
}

vector<OutputSentence*> StandardOutputReader::Read(const string& filename) {
  vector<LinearSentence*> corpus = ReadStandardSentences(filename, vocab, true);
  return vector<OutputSentence*>(corpus.begin(), corpus.end());
}

MorphologyOutputReader::MorphologyOutputReader() {}
MorphologyOutputReader::MorphologyOutputReader(const string& vocab_file, const string& root_vocab_file) {
  word_vocab.Convert("UNK");
  word_vocab.Convert("<s>");
  word_vocab.Convert("</s>");
  root_vocab.Convert("UNK");
  root_vocab.Convert("<s>");
  root_vocab.Convert("</s>");
  if (vocab_file.length() > 0) {
    ReadDict(vocab_file, word_vocab);
    word_vocab.Freeze();
    word_vocab.SetUnk("UNK");
  }

  if (root_vocab_file.length() > 0) {
    ReadDict(root_vocab_file, root_vocab);
    root_vocab.Freeze();
    root_vocab.SetUnk("UNK");
  }
}

vector<OutputSentence*> MorphologyOutputReader::Read(const string& filename) {
  vector<LinearSentence*> corpus = ReadMorphologySentences(filename, word_vocab, root_vocab, affix_vocab, char_vocab, true);
  return vector<OutputSentence*>(corpus.begin(), corpus.end());
}

vector<OutputSentence*> RnngOutputReader::Read(const string& filename) {
  vector<LinearSentence*> corpus = ReadStandardSentences(filename, vocab, false);
  return vector<OutputSentence*>(corpus.begin(), corpus.end());
}

string StandardOutputReader::ToString(const Word* word) {
  assert (false);
}

string MorphologyOutputReader::ToString(const Word* word) {
  assert (false);
}

string RnngOutputReader::ToString(const Word* word) {
  assert (false);
}

void ReadDict(const string& filename, Dict& dict) {
  ifstream f(filename);

  for (string line; getline(f, line);) {
    dict.Convert(strip(line));
  }
}

Bitext ReadBitext(const string& source_filename, const string& target_filename, InputReader* input_reader, OutputReader* output_reader) {
  vector<InputSentence*> source = input_reader->Read(source_filename);
  vector<OutputSentence*> target = output_reader->Read(target_filename);
  assert (source.size() == target.size());

  Bitext bitext;
  for (unsigned i = 0; i < source.size(); ++i) {
    bitext.push_back(make_pair(source[i], target[i]));
  }
  return bitext;
}

void Serialize(const InputReader* const input_reader, const OutputReader* const output_reader, const Translator& translator, Model& cnn_model) {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {}
  fseek(stdout, 0, SEEK_SET);

  boost::archive::binary_oarchive oa(cout);
  oa & cnn_model;
  oa & input_reader;
  oa & output_reader;
  oa & translator;
}

void Deserialize(const string& filename, InputReader*& input_reader, OutputReader*& output_reader, Translator& translator, Model& cnn_model) {
  ifstream f(filename);
  boost::archive::binary_iarchive ia(f);
  ia & cnn_model;
  ia & input_reader;
  ia & output_reader;
  ia & translator;
  f.close();
}


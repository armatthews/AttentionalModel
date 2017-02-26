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
    r->push_back(make_shared<StandardWord>(dict.convert("<s>")));
  }
  for (const string& w : words) {
    r->push_back(make_shared<StandardWord>(dict.convert(w)));
  }
  if (add_bos_eos) {
    r->push_back(make_shared<StandardWord>(dict.convert("</s>")));
  }
  return r;
}

vector<LinearSentence*> ReadStandardSentences(const string& filename, Dict& dict, bool add_bos_eos) {
  ifstream f(filename);
  if (!f.is_open()) {
    cerr << "Unable to open " << filename << " for reading." << endl;
    assert (f.is_open());
  }

  vector<LinearSentence*> sentences;
  for (string line; getline(f, line);) {
    LinearSentence* sentence = ReadStandardSentence(strip(line), dict, add_bos_eos);
    sentences.push_back(sentence);
  }
  return sentences;
}

shared_ptr<MorphoWord> ReadMorphoWord(const string& line, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab) {
  shared_ptr<MorphoWord> word = make_shared<MorphoWord>();
  vector<string> parts = tokenize(strip(line), "\t");
  string& word_str = parts[0];
  word->word = word_vocab.convert(word_str);

  for (unsigned i = 1; i < parts.size(); ++i) {
    vector<string> morphemes = tokenize(parts[i], "+");
    assert (morphemes.size() > 0);
    Analysis analysis;
    analysis.root = root_vocab.convert(morphemes[0]);
    for (unsigned j = 1; j < morphemes.size(); ++j) {
      analysis.affixes.push_back(affix_vocab.convert(morphemes[j]));
    }
    word->analyses.push_back(analysis);
  }

  for (unsigned i = 0; i < word_str.length(); ) {
    unsigned len = UTF8Len(parts[0][i]);
    string c = parts[0].substr(i, len);
    word->chars.push_back(char_vocab.convert(c));
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
  if (!f.is_open()) {
    cerr << "Unable to open " << filename << " for reading." << endl;
    assert (f.is_open());
  }

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

void StandardInputReader::Freeze() {
  if (!vocab.is_frozen()) {
    vocab.freeze();
    vocab.set_unk("UNK");
  }
}

vector<InputSentence*> SyntaxInputReader::Read(const string& filename) {
  ifstream f(filename);
  if (!f.is_open()) {
    cerr << "Unable to open " << filename << " for reading." << endl;
    assert (f.is_open());
  }

  vector<InputSentence*> sentences;
  for (string line; getline(f, line);) {
    SyntaxTree* sentence = new SyntaxTree(strip(line), &terminal_vocab, &nonterminal_vocab);
    sentence->AssignNodeIds();
    sentences.push_back(sentence);
  }

  return sentences;
}

void SyntaxInputReader::Freeze() {
  if (!terminal_vocab.is_frozen()) {
    terminal_vocab.freeze();
    terminal_vocab.set_unk("UNK");
  }
  if (!nonterminal_vocab.is_frozen()) {
    nonterminal_vocab.freeze();
    nonterminal_vocab.set_unk("UNK");
  }
}

vector<InputSentence*> MorphologyInputReader::Read(const string& filename) {
  vector<LinearSentence*> corpus = ReadMorphologySentences(filename, word_vocab, root_vocab, affix_vocab, char_vocab, true);
  return vector<InputSentence*>(corpus.begin(), corpus.end());
}

void MorphologyInputReader::Freeze() {
  if (!word_vocab.is_frozen()) {
    word_vocab.freeze();
    word_vocab.set_unk("UNK");
    root_vocab.freeze();
    root_vocab.set_unk("UNK");
    affix_vocab.freeze();
    affix_vocab.set_unk("UNK");
    char_vocab.freeze();
    char_vocab.set_unk("UNK");
  }
}

StandardOutputReader::StandardOutputReader() {}
StandardOutputReader::StandardOutputReader(const string& vocab_file) {
  vocab.convert("UNK");
  vocab.convert("<s>");
  vocab.convert("</s>");
  if (vocab_file.length() > 0) {
    ReadDict(vocab_file, vocab);
    vocab.freeze();
    vocab.set_unk("UNK");
  }
}

vector<OutputSentence*> StandardOutputReader::Read(const string& filename) {
  vector<LinearSentence*> corpus = ReadStandardSentences(filename, vocab, true);
  return vector<OutputSentence*>(corpus.begin(), corpus.end());
}

void StandardOutputReader::Freeze() {
  if (!vocab.is_frozen()) {
    vocab.freeze();
    vocab.set_unk("UNK");
  }
}

MorphologyOutputReader::MorphologyOutputReader() {}
MorphologyOutputReader::MorphologyOutputReader(const string& vocab_file, const string& root_vocab_file) {
  word_vocab.convert("UNK");
  word_vocab.convert("<s>");
  word_vocab.convert("</s>");
  root_vocab.convert("UNK");
  root_vocab.convert("<s>");
  root_vocab.convert("</s>");
  if (vocab_file.length() > 0 && !word_vocab.is_frozen()) {
    ReadDict(vocab_file, word_vocab);
    word_vocab.freeze();
    word_vocab.set_unk("UNK");
  }

  if (root_vocab_file.length() > 0 && !root_vocab.is_frozen()) {
    ReadDict(root_vocab_file, root_vocab);
    root_vocab.freeze();
    root_vocab.set_unk("UNK");
  }
}

vector<OutputSentence*> MorphologyOutputReader::Read(const string& filename) {
  vector<LinearSentence*> corpus = ReadMorphologySentences(filename, word_vocab, root_vocab, affix_vocab, char_vocab, true);
  return vector<OutputSentence*>(corpus.begin(), corpus.end());
}

void MorphologyOutputReader::Freeze() {
  if (!word_vocab.is_frozen()) {
    word_vocab.freeze();
    word_vocab.set_unk("UNK");
    root_vocab.freeze();
    root_vocab.set_unk("UNK");
    affix_vocab.freeze();
    affix_vocab.set_unk("UNK");
    char_vocab.freeze();
    char_vocab.set_unk("UNK");
  }
}

vector<OutputSentence*> RnngOutputReader::Read(const string& filename) {
  vector<LinearSentence*> corpus = ReadStandardSentences(filename, vocab, false);
  return vector<OutputSentence*>(corpus.begin(), corpus.end());
}

void RnngOutputReader::Freeze() {
  if (!vocab.is_frozen()) {
    vocab.freeze();
  }
}

string StandardOutputReader::ToString(const shared_ptr<const Word> word) {
  const shared_ptr<const StandardWord> w = dynamic_pointer_cast<const StandardWord>(word);
  return vocab.convert(w->id);
}

string MorphologyOutputReader::ToString(const shared_ptr<const Word> word) {
  assert (false);
}

string RnngOutputReader::ToString(const shared_ptr<const Word> word) {
  const shared_ptr<const StandardWord> w = dynamic_pointer_cast<const StandardWord>(word);
  return vocab.convert(w->id);
}

void ReadDict(const string& filename, Dict& dict) {
  ifstream f(filename);

  for (string line; getline(f, line);) {
    dict.convert(strip(line));
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

Bitext ReadBitextWithTrees(const string& source_filename, const string& source_tree_filename, const string& target_filename, InputReader* input_reader, InputReader* tree_reader, OutputReader* output_reader) {
  cerr << "Source filename: " << source_filename << endl;
  cerr << "Tree filename: " << source_tree_filename << endl;
  vector<InputSentence*> source = input_reader->Read(source_filename);
  vector<InputSentence*> trees = tree_reader->Read(source_tree_filename);
  vector<OutputSentence*> target = output_reader->Read(target_filename);
  assert (source.size() == target.size());

  Bitext bitext;
  for (unsigned i = 0; i < source.size(); ++i) {
    SyntaxTree* tree = dynamic_cast<SyntaxTree*>(trees[i]);
    InputSentence* sent_with_tree = new SentWithTree(source[i], tree);
    if (i == 0) {
      cerr << "First sentence has " << tree->NumNodes() << " tree nodes and " << source[i]->NumNodes() << " linear nodes" << endl;
    }
    bitext.push_back(make_pair(sent_with_tree, target[i]));
  }
  return bitext;
}

void Serialize(const InputReader* const input_reader, const OutputReader* const output_reader, const Translator& translator, Model& dynet_model, const Trainer* const trainer) {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {}
  fseek(stdout, 0, SEEK_SET);

  boost::archive::binary_oarchive oa(cout);
  oa & dynet_model;
  oa & input_reader;
  oa & output_reader;
  oa & translator;
  oa & trainer;
}

void Deserialize(const string& filename, InputReader*& input_reader, OutputReader*& output_reader, Translator& translator, Model& dynet_model, Trainer*& trainer) {
  ifstream f(filename);
  boost::archive::binary_iarchive ia(f);
  ia & dynet_model;
  ia & input_reader;
  ia & output_reader;
  ia & translator;
  ia & trainer;
  f.close();
}

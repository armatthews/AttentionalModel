#include "dynet/dynet.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include "io.h"
#include "kbestlist.h"
#include "utils.h"

using namespace dynet;
using namespace std;
namespace po = boost::program_options;

vector<Expression> encodings;
RnngOutputReader* g_output_reader;
RnngOutputModel* g_output_model;
Translator* g_translator;

vector<RNNPointer> ps;
vector<Expression> ss;
vector<Expression> cs;

unsigned AddWord(unsigned parent_index, string word) {
    RNNPointer parentp = ps[parent_index];
    Expression parents = ss[parent_index];
    Expression parentc = cs[parent_index];

    StandardWord* w = new StandardWord(g_output_reader->vocab.convert(word));
    cout << "Loss(" << (int)parentp << ", " << g_output_reader->vocab.convert(w->id) << ") = " << as_scalar(g_output_model->Loss(parentp, parents, w).value()) << endl;
    g_output_model->AddInput(w, parentc, parentp);

    RNNPointer newp = g_output_model->GetStatePointer();
    Expression news = g_output_model->GetState(newp);
    Expression newc = g_translator->attention_model->GetContext(encodings, news);

    ps.push_back(newp);
    ss.push_back(news);
    cs.push_back(newc);
    return ps.size() - 1;
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  const string model_filename = "/tmp/btec_easy_rnng2.out";
  const unsigned beam_size = 1;
  const unsigned max_length = 100;
  const unsigned kbest_size = 1;

  InputReader* input_reader_base = nullptr;
  OutputReader* output_reader_base = nullptr;
  Model dynet_model;
  Translator translator;
  Trainer* trainer = nullptr;
  Deserialize(model_filename, input_reader_base, output_reader_base, translator, dynet_model, trainer);
  StandardInputReader* input_reader = dynamic_cast<StandardInputReader*>(input_reader_base);
  RnngOutputReader* output_reader = dynamic_cast<RnngOutputReader*>(output_reader_base);
  translator.SetDropout(0.0f);

  g_output_reader = output_reader; // XXX
  g_translator = &translator; // XXX

  vector<InputSentence*> source_sentences = input_reader->Read("btec_easy/dev.zh");
  vector<OutputSentence*> target_sentences = output_reader->Read("/tmp/dev.en.rnng");
  LinearSentence& source = *dynamic_cast<LinearSentence*>(source_sentences[0]);
  OutputSentence& target = *target_sentences[0];

  RnngOutputModel* output_model = dynamic_cast<RnngOutputModel*>(translator.output_model);
  SourceConditionedParserBuilder* builder = output_model->builder;

  g_output_model = output_model; // XXX

  ComputationGraph cg;
  translator.NewGraph(cg);

  encodings = translator.encoder_model->Encode(&source);
  translator.attention_model->NewSentence(&source);
  KBestList<Word*> kbest_next;

  // Initial state
  RNNPointer p0 = output_model->GetStatePointer();
  Expression s0 = output_model->GetState(p0);
  Expression c0 = translator.attention_model->GetContext(encodings, s0);

  ps.push_back(p0); ss.push_back(s0); cs.push_back(c0);

  if (false) {
    if (argc == 1) {
      cerr << "Breaking things..." << endl;
      // Node 0 --> Node 1, NT(S)
      StandardWord* w1 = new StandardWord(output_reader->vocab.convert("NT(S)"));
      cout << "Loss(" << (int)p0 << ", " << output_reader->vocab.convert(w1->id) << ") = " << as_scalar(output_model->Loss(p0, s0, w1).value()) << endl;
      output_model->AddInput(w1, c0, p0);

      RNNPointer p1 = output_model->GetStatePointer();
      Expression s1 = output_model->GetState(p1);
      Expression c1 = translator.attention_model->GetContext(encodings, s1);
    }
    else {
      cerr << "Not breaking things..." << endl;
    }

    // Node 0 --> Node 2, NT(SQ)
    StandardWord* w2 = new StandardWord(output_reader->vocab.convert("NT(SQ)"));
    cout << "Loss(" << (int)p0 << ", " << output_reader->vocab.convert(w2->id) << ") = " << as_scalar(output_model->Loss(p0, s0, w2).value()) << endl;
    Expression s2 = output_model->AddInput(w2, c0, p0);

    RNNPointer p2 = output_model->GetStatePointer();
    //Expression s2 = output_model->GetState(p2);
    cerr << "        s2 is: " << vec2str(s2) << endl;
    cerr << "    should be: " << "0 0 0 0 0 0 0 0 0 0.140581 0 0 0 0 0.247572 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.350066 0 0.0113835 0 0 0 0.0707398 0 0.0739977 0.26049 0 0 0 0 0.599113 0 0 0 0.0519224 0.0610349 0.457624" << endl;
    cerr << "should NOT be: " << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.726698 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0992733 0 0 0 0 0 0 0 0 0.467469 0 0 0 0 0.543195 0 0 0 0 0 0" << endl;
    Expression c2 = translator.attention_model->GetContext(encodings, s2);
  }
  else {
    unsigned i0 = 0;
    unsigned i1 = AddWord(i0, "NT(S)");
    unsigned i2 = AddWord(i0, "NT(SQ)");
    unsigned i3 = AddWord(i0, "NT(SBARQ)");

    unsigned i4 = AddWord(i1, "NT(NP)");
    unsigned i5 = AddWord(i1, "NT(VP)");
    unsigned i6 = AddWord(i1, "NT(S)");

    unsigned i7 = AddWord(i2, "SHIFT(is)");
    unsigned i8 = AddWord(i2, "SHIFT(does)");
    unsigned i9 = AddWord(i2, "SHIFT(will)");

    unsigned i10 = AddWord(i3, "NT(WHNP)");
    //unsigned i11 = AddWord(i3, "NT(WHADVP)");
    //unsigned i12 = AddWord(i3, "NT(NP)");

    //unsigned i13 = AddWord(i4, "SHIFT(it)");
    //unsigned i14 = AddWord(i4, "SHIFT(something)");
    //unsigned i15 = AddWord(i4, "SHIFT(that)");

    //unsigned i16 = AddWord(i5, "SHIFT(UNK)");
    //unsigned i17 = AddWord(i5, "NT(NP)");
    //unsigned i18 = AddWord(i5, "NT(PP)");

    unsigned i19 = AddWord(i10, "SHIFT(what)");
    cout << "       state_vector should be: 0.249708 0 0 0 0 0 0.28125 0.0946464 0 0 0 0 0 0 0 0 0 0 0 0.204591 0 0.254825 0 0 0 0.711557 0 0.359613 0.542675 0 0.0343341 0 0 0 0.572632 0 0 0 0 0 0 0 0 0 0 0 0.0274155 0.384143 0 0 0 0 0 0 0 0.048484 0 0 0 0 0 0 0.0558983 0" << endl;
    //unsigned i20 = AddWord(i10, "SHIFT(that)");
    //unsigned i21 = AddWord(i10, "SHIFT(i)");

    /*KBestList<Word*> kbest = output_model->PredictKBest(ps[i10], ss[i10], 10);
    for (auto& p : kbest.hypothesis_list()) {
      cout << output_reader->vocab.convert(dynamic_cast<StandardWord*>(get<1>(p))->id) << "\t" << get<0>(p) << endl;
    }*/
  }

  return 0;
}

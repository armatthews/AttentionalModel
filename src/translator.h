#pragma once
#include <boost/serialization/access.hpp>
#include "encoder.h"
#include "attention.h"
#include "output.h"

template <class Input>
class Translator {
public:
  Translator() {}
  Translator(EncoderModel<Input>* encoder, AttentionModel* attention, OutputModel* output);
  Expression BuildGraph(const Input& source, const Sentence& target, ComputationGraph& cg);
  void InitializeParameters(Model& model) {
    cerr << "Initializing Translator" << endl;
    encoder_model->InitializeParameters(model);
    attention_model->InitializeParameters(model);
    output_model->InitializeParameters(model);
  }

private:
  EncoderModel<Input>* encoder_model;
  AttentionModel* attention_model;
  OutputModel* output_model;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int);
};

template<class Input>
Translator<Input>::Translator(EncoderModel<Input>* encoder, AttentionModel* attention, OutputModel* output) {
  encoder_model = encoder;
  attention_model = attention;
  output_model = output;
}

template <class Input>
Expression Translator<Input>::BuildGraph(const Input& source, const Sentence& target, ComputationGraph& cg) {
  encoder_model->NewGraph(cg);
  attention_model->NewGraph(cg);
  output_model->NewGraph(cg);

  vector<Expression> word_losses(target.size());
  word_losses[0] = input(cg, 0.0f); // <s>

  for (unsigned i = 1; i < target.size(); ++i) {
    const WordId& prev_word = target[i - 1];
    const WordId& curr_word = target[i];
    vector<Expression> encodings = encoder_model->Encode(source);
    Expression state = output_model->GetState();
    Expression context = attention_model->GetContext(encodings, state);
    word_losses[i] = output_model->Loss(prev_word, context, curr_word);
  }
  return sum(word_losses);
}

template<class Input>
template<class Archive>
void Translator<Input>::serialize(Archive& ar, const unsigned int) {
  cerr << "serializing translator" << endl;
  ar & encoder_model;
  ar & attention_model;
  ar & output_model;
}

#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <fstream>
#include <csignal>

#include "bitext.h"
#include "attentional.h"
#include "decoder.h"
#include "utils.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

float dot(vector<float> a, vector<float> b) {
  assert (a.size() == b.size());
  float d = 0.0;
  for (unsigned i = 0; i < a.size(); ++i) {
    d += a[i] * b[i];
  }
  return d;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: cat source.txt | " << argv[0] << " model" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);

  const string model_filename = argv[1];

  cnn::Initialize(argc, argv);

  Dict source_vocab;
  Dict target_vocab;
  Model* cnn_model;
  AttentionalModel* attentional_model;
  cerr << "Loading model..." << endl;
  tie(source_vocab, target_vocab, cnn_model, attentional_model) = LoadModel(model_filename);

  assert (source_vocab.Contains("<s>"));
  assert (source_vocab.Contains("</s>"));
  source_vocab.Freeze();
  target_vocab.Freeze();
  cerr << "Done!" << endl;

  string line;
  getline(cin, line);
  vector<string> words = tokenize(line, " ");

  vector<WordId> source;
  for (string w : words) {
    WordId id = source_vocab.Convert(w);
    source.push_back(id);
  }

  ComputationGraph cg;
  vector<Expression> annotations;
  Expression zeroth_context;
  tie(annotations, zeroth_context) = attentional_model->BuildAnnotationVectors(source, cg);
  assert (annotations.size() == source.size());
  cg.forward();

  vector<vector<float>> annotation_vectors; 
  for (unsigned i = 0; i < annotations.size(); ++i) {
    const Tensor& t = annotations[i].value();
    const vector<float> v = as_vector(t);
    annotation_vectors.push_back(v);
  }

  cout << "Annotation similarities:" << endl;
  for (unsigned i = 0; i < source.size(); ++i) {
    for (unsigned j = i + 1; j < source.size(); ++j) {
      cout << source_vocab.Convert(source[i]) << "\t" << source_vocab.Convert(source[j]) << "\t";
      cout << dot(annotation_vectors[i], annotation_vectors[j]) << endl;
    }
  }
  cerr << endl;

  const unsigned source_size = source.size();

  attentional_model->output_builder.new_graph(cg);
  attentional_model->output_builder.start_new_sequence();

  MLP aligner = attentional_model->GetAligner(cg);
  unsigned t = 1;
  Expression prev_context = zeroth_context;
  Expression prev_target_word_embedding = lookup(cg, attentional_model->p_Et, source[t - 1]);
  const RNNPointer& rnn_pointer = attentional_model->output_builder.state();

  Expression annotation_matrix = concatenate_cols(annotations);
  Expression state_rnn_input = concatenate({prev_context, prev_target_word_embedding});
  Expression new_state = attentional_model->output_builder.add_input(rnn_pointer, state_rnn_input);

  Expression new_bias = affine_transform({aligner.i_Hb, aligner.i_IH[0], new_state});
  Expression bias_broadcast1 = concatenate_cols(vector<Expression>(annotations.size(), new_bias));
  Expression hidden = tanh(affine_transform({bias_broadcast1, aligner.i_IH[1], annotation_matrix}));
  Expression bias_broadcast2 = concatenate_cols(vector<Expression>(annotations.size(), aligner.i_Ob));
  Expression unnormalized_alignment_vector = transpose(affine_transform({bias_broadcast2, aligner.i_HO, hidden}));

  Expression normalized_alignment_vector = softmax(unnormalized_alignment_vector);
  Expression first_context = annotation_matrix * normalized_alignment_vector;

  MLP final_mlp = attentional_model->GetFinalMLP(cg);
  Expression output_dist = final_mlp.Feed({prev_target_word_embedding, new_state, first_context});

  cg.incremental_forward();

  cout << "Alignment probs:" << endl;
  vector<float> norm_align_v = as_vector(normalized_alignment_vector.value());
  for (float j : norm_align_v) {
    cout << j << " ";
  }
  cout << "\b\n";
  cout << endl;

  cout << "Zeroth context:" << endl;
  for (float j : as_vector(zeroth_context.value())) {
    cout << j << " ";
  }
  cout << "\b\n" << endl;

  cout << "First context:" << endl;
  for (float j : as_vector(first_context.value())) {
    cout << j << " ";
  }
  cout << "\b\n" << endl;

  cout << "Output state:" << endl;
  for (float j : as_vector(new_state.value())) {
    cout << j << " ";
  }
  cout << "\b\n" << endl;

  cout << "Output distribution:" << endl;
  vector<float> output_dist_v = as_vector(output_dist.value());
  assert (output_dist_v.size() == target_vocab.size());
  for (unsigned i = 0; i < output_dist_v.size(); ++i) {
    cout << target_vocab.Convert(i) << "\t" << output_dist_v[i] << endl;
  }

  return 0;
}

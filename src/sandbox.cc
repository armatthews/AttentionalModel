#include "cnn/dict.h"
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/lstm.h"
#include "utils.h"
#include "kbestlist.h"
#include "syntax_tree.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/algorithm/string.hpp>

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <climits>
#include <csignal>

#define NONLINEAR
#define FAST

using namespace std;
using namespace cnn;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

int test() {
  KBestList<string> kbest(3);
  //kbest.add(5.0, "hello");
  //kbest.add(4.0, "salut");
  //kbest.add(6.0, "konnichiwa");
  kbest.add(-1.0, "nihao");
  kbest.add(16.0, "hola");

  for (auto p : kbest.hypothesis_list()) {
    cout << p.first << "\t" << p.second << endl;
  }
}

int main(int argc, char** argv) {
  string line;
  Dict dict;
  while (getline(cin, line)) {
    boost::algorithm::trim(line);
    SyntaxTree tree(line, &dict);
    cout << tree << endl;
    //cout << tree.MaxBranchCount() << endl;
  }
  return 0;
  cerr << SIZE_MAX << endl;
  return test();
  signal (SIGINT, ctrlc_handler);

  cnn::Initialize(argc, argv);
  Model model;
  SimpleSGDTrainer sgd(&model);

  const unsigned INPUT_SIZE = 2;
  const unsigned OUTPUT_SIZE = 2;
  vector<cnn::real> x_values = {1, 4, 9, 16};
  vector<cnn::real> y_values = {17.2, 33.2};

  vector<cnn::real> bias = {5, 7, 5, 7};
  vector<cnn::real> weights = {1, 3, 2, 4};
  vector<cnn::real> alpha = {0.9, 0.1};

  Parameters& p_W = *model.add_parameters(Dim(INPUT_SIZE, OUTPUT_SIZE));
  Parameters& p_b = *model.add_parameters(Dim(OUTPUT_SIZE, 2));
  Parameters& p_a = *model.add_parameters(Dim(OUTPUT_SIZE));

  cerr << "Training model...\n";
  for (unsigned iteration = 0; iteration < 1000; iteration++) {
    ComputationGraph hg;
    VariableIndex i_W = hg.add_parameters(&p_W);
    //VariableIndex i_W = hg.add_input(p_W.dim, &weights);
    //VariableIndex i_b = hg.add_input(p_b.dim, &bias);
    VariableIndex i_b = hg.add_parameters(&p_b);
    VariableIndex i_a = hg.add_input(p_a.dim, &alpha);

    VariableIndex i_x = hg.add_input({INPUT_SIZE, 2}, &x_values);
    VariableIndex i_y = hg.add_input({OUTPUT_SIZE}, &y_values);
    VariableIndex i_t = hg.add_function<AffineTransform>({i_b, i_W, i_x});
    VariableIndex i_yhat = hg.add_function<SumColumns>({i_t, i_a});

    const Tensor& t = hg.incremental_forward();
    for (unsigned i = 0; i < t.d.rows(); ++i) {
      cout << (i == 0 ? "" : "\n");
      for (unsigned j = 0; j < t.d.cols(); ++j) {
        cout << (j == 0 ? "" : " ") << TensorTools::AccessElement(t, Dim(i, j));
      }
    }
    cout << endl;
    VariableIndex i_d = hg.add_function<SquaredEuclideanDistance>({i_y, i_yhat});
    double loss = as_scalar(hg.forward()); 
    if (ctrlc_pressed) {
      break;
    }
    hg.backward();
    sgd.update(0.1);
    cout << "Iteration " << iteration << " loss: " << loss << endl;
    //sgd.update_epoch();
  }

  cout << "Weight matrix:" << endl;
  cout << TensorTools::AccessElement(p_W.values, Dim(0, 0)) << " "
       <<  TensorTools::AccessElement(p_W.values, Dim(1, 0)) << endl;
  cout << TensorTools::AccessElement(p_W.values, Dim(0, 1)) << " "
       <<  TensorTools::AccessElement(p_W.values, Dim(1, 1)) << endl;

  cout << "bias matrix:" << endl;
  cout << TensorTools::AccessElement(p_b.values, Dim(0, 0)) << " "
       <<  TensorTools::AccessElement(p_b.values, Dim(1, 0)) << endl;
  cout << TensorTools::AccessElement(p_b.values, Dim(0, 1)) << " "
       <<  TensorTools::AccessElement(p_b.values, Dim(1, 1)) << endl;

  return 0;
}

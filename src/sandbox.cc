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

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);

  cnn::Initialize(argc, argv);

  ComputationGraph cg;
  vector<cnn::real> matrix_values = {1, 2, 3, 4};
  Expression matrix = input(cg, {2, 2}, &matrix_values);
  Expression th = tanh(matrix);
  //Expression sm = softmax(matrix);

  const Tensor& output = cg.forward();
  cout << output.d << endl;
  assert (output.d.nd <= 2);
  if (output.d.nd == 0) {
    cout << "(Output is 0-dimensional)" <<endl;
  }
  else if (output.d.nd == 1) {
    for (unsigned i = 0; i < output.d[0]; ++i) {
      cout << TensorTools::AccessElement(output, {i}) << " ";
    }
    cout << "\b" << endl;
  }
  else if (output.d.nd == 2) {
    for (unsigned i = 0; i < output.d[1]; ++i) {
      for (unsigned j = 0; j < output.d[0]; ++j) {
        cout << TensorTools::AccessElement(output, {i, j}) << " ";
      }
      cout << "\b" << endl;
    }
  }
  return 0;
}

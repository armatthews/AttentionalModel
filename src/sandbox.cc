#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/join.hpp>

#include <sys/types.h>
#include <iostream>
#include <sys/wait.h>
#include <fstream>
#include <csignal>

#include "bitext.h"
#include "attentional.h"
#include "decoder.h"
#include "utils.h"

using namespace cnn;
using namespace std;

int main(int argc, char** argv) {

  cnn::Initialize(argc, argv, 1, true);

  Model* cnn_model = new Model();
  Parameters* p = cnn_model->add_parameters({100000});
  pid_t pid = fork();
  if (pid != 0) { // parent
    sleep(5);
    for (unsigned i = 0; i < 10; ++i) {
      vector<float> v(100000);
      for (unsigned j = 0; j < 100000; ++j) {
        v[j] = 1.0 * rand() / RAND_MAX * 2.0 - 1.0;
      }
      TensorTools::SetElements(p->values, v);
      sleep(3);
    }
    wait(NULL);
  }
  else {
    // child
    for (unsigned i = 0; i < 60; ++i) {
      cout << TensorTools::AccessElement(p->values, {0, 0}) << endl;
      sleep(1);
    }
  }
  return 0;
}

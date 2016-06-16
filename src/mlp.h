#pragma once
#include <boost/serialization/access.hpp>
#include "cnn/cnn.h"
#include "cnn/expr.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class MLP {
public:
  MLP();
  MLP(Model& model, unsigned input_size, unsigned hidden_size, unsigned output_size);
  void NewGraph(ComputationGraph& cg);
  Expression Feed(Expression input) const;

private:
  Parameter p_wIH;
  Parameter p_wHO;
  Parameter p_wHb;
  Parameter p_wOb;

  Expression wIH;
  Expression wHO;
  Expression wHb;
  Expression wOb;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & p_wIH;
    ar & p_wHO;
    ar & p_wHb;
    ar & p_wOb;
  }
};

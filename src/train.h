#pragma once
#include "dynet/dynet.h"
#include "dynet/mp.h"
#include <boost/program_options.hpp>

#include <iostream>
#include <csignal>
#include "syntax_tree.h"
#include "translator.h"
#include "tree_encoder.h"
#include "io.h"
#include "utils.h"

using namespace dynet;
using namespace dynet::expr;
using namespace dynet::mp;
using namespace std;
namespace po = boost::program_options;

enum InputType {kStandard = 0, kSyntaxTree = 1, kMorphology = 2, kRNNG = 3, kDependency = 4};
istream& operator>>(istream& in, InputType& input_type);

void AddTrainerOptions(po::options_description& desc);
Trainer* CreateTrainer(Model& dynet_model, const po::variables_map& vm);

InputReader* CreateInputReader(const po::variables_map& vm);
OutputReader* CreateOutputReader(const po::variables_map& vm);
unsigned ComputeAnnotationDim(const po::variables_map& vm);
EncoderModel* CreateEncoderModel(const po::variables_map& vm, Model& dynet_model, InputReader* input_reader);
void AddPriors(const po::variables_map& vm, AttentionModel* attention_model, Model& dynet_model);
AttentionModel* CreateAttentionModel(const po::variables_map& vm, Model& dynet_model);
OutputModel* CreateOutputModel(const po::variables_map& vm, Model& dynet_model, OutputReader* output_reader);

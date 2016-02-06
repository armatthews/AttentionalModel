#include "tree_encoder.h"
BOOST_CLASS_EXPORT_IMPLEMENT(TreeEncoder)

const unsigned lstm_layer_count = 2;

TreeEncoder::TreeEncoder() {}

TreeEncoder::TreeEncoder(Model& model, unsigned vocab_size, unsigned label_vocab_size, unsigned input_dim, unsigned output_dim)
  : vocab_size(vocab_size), label_vocab_size(label_vocab_size), input_dim(input_dim), output_dim(output_dim) {
  InitializeParameters(model);
}

void TreeEncoder::InitializeParameters(Model& model) {
  tree_builder = new SocherTreeLSTMBuilder(5, lstm_layer_count, output_dim, output_dim, &model);
  label_embeddings = model.add_lookup_parameters(label_vocab_size, {output_dim});
  linear_encoder = new BidirectionalSentenceEncoder(model, vocab_size, input_dim, output_dim);
}

void TreeEncoder::NewGraph(ComputationGraph& cg) {
  linear_encoder->NewGraph(cg);
  tree_builder->new_graph(cg);
  pcg = &cg;
}

vector<Expression> TreeEncoder::Encode(const TranslatorInput* const input) {
  const SyntaxTree& sentence = *dynamic_cast<const SyntaxTree*>(input);
  Sentence terminals = sentence.GetTerminals();
  vector<Expression> linear_encodings = linear_encoder->Encode(&terminals);
  tree_builder->start_new_sequence();

  vector<Expression> node_encodings;
  vector<const SyntaxTree*> node_stack = {&sentence};
  vector<unsigned> index_stack = {0};
  unsigned terminal_index = 0;

  // We will build an encoding for each node in the SyntaxTree, bottom-up,
  // left to right. For each node, we first encode all of its children,
  // and then once that's done, use their encodings to produce an encoding
  // for the current node.
  while (node_stack.size() > 0) {
    assert (node_stack.size() == index_stack.size());
    const SyntaxTree* node = node_stack.back();
    const unsigned i = index_stack.back();
    if (i < node->NumChildren()) {
      // The current node still has children,
      // so push the next one onto the stack.
      index_stack[index_stack.size() - 1] += 1;
      node_stack.push_back(&node->GetChild(i));
      index_stack.push_back(0);
    }
    else {
      // All our children are done. Create an embedding for this node.
      assert (node_encodings.size() == node->id());
      vector<int> children(node->NumChildren());
      for (unsigned j = 0; j < node->NumChildren(); ++j) {
        unsigned child_id = node->GetChild(j).id();
        assert (child_id < node_encodings.size());
        children[j] = (int)child_id;
      }

      Expression input_expr;
      if (node->NumChildren() == 0) {
        // This is a terminal. Just use its linear encoding
        assert (terminal_index < linear_encodings.size());
        input_expr = linear_encodings[terminal_index++];
      }
      else {
        // If this is an NT, then its encoding will be built from
        // embeddings of its label and its children.
        input_expr = lookup(*pcg, label_embeddings, node->label());
      }
      Expression node_encoding = tree_builder->add_input((int)node->id(), children, input_expr);
      node_encodings.push_back(node_encoding);
      index_stack.pop_back();
      node_stack.pop_back();
    }
  }
  assert (node_stack.size() == index_stack.size());
  assert (node_stack.size() == 0);
  return node_encodings;
}


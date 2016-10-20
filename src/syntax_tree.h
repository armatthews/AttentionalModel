#pragma once
#include <vector>
#include <string>
#include <stack>
#include "dynet/dict.h"
#include "utils.h"

using namespace std;
using namespace dynet;

class SyntaxTree;

enum TreeIterationOrder {PreOrder, PostOrder};

class SyntaxTreeIterator {
public:
  SyntaxTreeIterator(SyntaxTree* root, TreeIterationOrder order);
  SyntaxTree& operator*();
  bool operator==(const SyntaxTreeIterator& other);
  bool operator!=(const SyntaxTreeIterator& other);
  SyntaxTreeIterator& operator++(); // pre-increment
  //SyntaxTreeIterator operator++(int); //post-increment
  SyntaxTree* node;
private:
  stack<SyntaxTree*> node_stack;
  stack<unsigned> index_stack;
  TreeIterationOrder order;
};

class SyntaxTree : public InputSentence {
public:
  SyntaxTree();
  SyntaxTree(string tree, Dict* word_dict, Dict* label_dict);

  bool IsTerminal() const;
  unsigned NumChildren() const;
  unsigned NumNodes() const;
  unsigned MaxBranchCount() const;
  unsigned MinDepth() const;
  unsigned MaxDepth() const;
  WordId label() const;
  unsigned id() const;
  LinearSentence GetTerminals() const;

  SyntaxTree& GetChild(unsigned i);
  const SyntaxTree& GetChild(unsigned i) const;

  string ToString() const;
  unsigned AssignNodeIds(unsigned start = 0);

  SyntaxTreeIterator begin(TreeIterationOrder order) const;
  SyntaxTreeIterator end() const;
  unsigned size() const;

private:
  Dict* word_dict;
  Dict* label_dict;
  WordId label_;
  unsigned id_;
  vector<SyntaxTree> children;
};

ostream& operator<< (ostream& stream, const SyntaxTree& tree);

#include <cassert>
#include <iostream>
#include <sstream>
#include "syntax_tree.h"

SyntaxTree::SyntaxTree() {}

SyntaxTree::SyntaxTree(string tree) {
  // Sometimes Berkeley parser fails to parse a sentence and just outputs ()
  if (tree == "()") {
    return;
  }

  // TODO: Handle terminals with ( or ) in them?

  // If we have a terminal
  if (tree.length() == 0 || tree[0] != '(') {
    assert (tree.find("(") == string::npos);
    assert (tree.find(")") == string::npos);
    assert (tree.find(" ") == string::npos);
    label_ = tree;
  }
  else {
    assert (tree[tree.length() - 1] == ')');
    unsigned first_space = tree.find(" ");
    assert (first_space != string::npos);
    label_ = tree.substr(1, first_space - 1);

    vector<string> child_strings;
    unsigned start = first_space + 1;
    unsigned i = start;
    unsigned open_parens = 0;
    for (; i < tree.length() - 1; ++i) {
      char c = tree[i];
      if (c == '(') {
        open_parens++;
      }
      else if (c == ')') {
        open_parens--;
        if (open_parens == 0) {
          child_strings.push_back(tree.substr(start, i - start + 1));
          start = i + 1;
        }
      }
      else if (c == ' ' && open_parens == 0) {
        if (i > start) {
          child_strings.push_back(tree.substr(start, i - start + 1));
       }
        start = i + 1;
      } 
    }

    unsigned end = tree.length() - 2;
    if (end >= start) {
      child_strings.push_back(tree.substr(start, end - start + 1));
    }

    for (string child_string : child_strings) {
      children.push_back(SyntaxTree(child_string));
    }
    assert (children.size() > 0);
  }
}

bool SyntaxTree::IsTerminal() const {
  return children.size() == 0;
}

unsigned SyntaxTree::NumChildren() const {
  return children.size();
}

unsigned SyntaxTree::NumNodes() const {
  unsigned node_count = 1;
  for (const SyntaxTree& child : children) {
    node_count += child.NumNodes();
  }
  return node_count;
}

unsigned SyntaxTree::MaxBranchCount() const {
  unsigned max_branch_count = children.size();
  for (const SyntaxTree& child : children) {
    unsigned n = child.MaxBranchCount();
    if (n > max_branch_count) {
      max_branch_count = n;
    }
  }
  return max_branch_count;
}

unsigned SyntaxTree::MinDepth() const {
  if (IsTerminal()) {
    return 0;
  }

  unsigned min_depth = children[0].MinDepth();
  for (unsigned i = 1; i < children.size(); ++i) {
    unsigned d = children[i].MinDepth();
    if (d < min_depth) {
      min_depth = d;
    }
  }

  return min_depth + 1;
}

unsigned SyntaxTree::MaxDepth() const {
  if (IsTerminal()) {
    return 0;
  }

  unsigned max_depth = children[0].MaxDepth();
  for (unsigned i = 1; i < children.size(); ++i) {
    unsigned d = children[i].MaxDepth();
    if (d > max_depth) {
      max_depth = d;
    }
  }

  return max_depth + 1;
}

SyntaxTree& SyntaxTree::GetChild(unsigned i) {
  return children[i];
}

const SyntaxTree& SyntaxTree::GetChild(unsigned i) const {
  return children[i];
}

string SyntaxTree::label() const {
  return label_;
}

string SyntaxTree::ToString() const {
  if (IsTerminal()) {
    return label_;
  }

  stringstream ss;
  ss << "(" << label_;
  for (const SyntaxTree& child : children) {
    ss << " " << child.ToString();
  }
  ss << ")";
  return ss.str();
}

ostream& operator<< (ostream& stream, const SyntaxTree& tree) {
  return stream << tree.ToString();
}


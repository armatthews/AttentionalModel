import sys

from collections import namedtuple
class SyntaxTreeNode:
  def __init__(self, label, parent, children):
    self.label = label
    self.parent = parent
    self.children = children

def parse_tree(line):
  structure = None
  curr_node = None

  i = 0
  while i < len(line):
    if line[i] == ' ':
      i += 1
      continue

    if line[i] == '(':
      # nt
      next_space = line.find(' ', i+1)
      nt = line[i+1:next_space]
      new_node = SyntaxTreeNode(nt, curr_node, [])
      if structure is None:
        structure = new_node
      else:
        curr_node.children.append(new_node)
      curr_node = new_node
      i = next_space + 1

    elif line[i] == ')':
      # reduce
      curr_node = curr_node.parent
      i += 1

    else:
      # terminal
      next_paren = line.find(')', i + 1)
      next_space = line.find(' ', i + 1)
      end = min(next_paren, next_space) if next_space >= 0 else next_paren
      term = line[i : end]
      curr_node.children.append(SyntaxTreeNode(term, curr_node, []))
      i = end

  return structure

def remove_unary_chains(tree):
  for i in range(len(tree.children)):
    while len(tree.children[i].children) == 1:
      tree.children[i] = tree.children[i].children[0]
  for child in tree.children:
    remove_unary_chains(child)

  if tree.parent == None:
    while len(tree.children) == 1 and len(tree.children[0].children) > 0:
      tree.children = tree.children[0].children

def tree_to_string(tree):
  if len(tree.children) == 0:
    return tree.label
  else:
    return '(%s %s)' % (tree.label, ' '.join(tree_to_string(child) for child in tree.children))

for line in sys.stdin:
  line = line.strip()
  if line.startswith('( ') and line.endswith(' )'):
    line = line[2:-2]
  tree = parse_tree(line.strip())
  remove_unary_chains(tree)
  print(tree_to_string(tree))

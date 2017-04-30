import sys
from collections import defaultdict

def get_oracle(arcs):
  children = defaultdict(list)
  for index, (word, head) in enumerate(arcs):
    children[head].append(index)

  open_nodes = [-1]
  child_indices = [0]
  out = []
  while len(open_nodes) > 0:
    assert len(open_nodes) == len(child_indices)
    parent = open_nodes[-1]
    child_index = child_indices[-1]
    prev_child = children[parent][child_index - 1] if (child_index > 0) else -1
    child_indices.pop()

    # We're done with this constit
    if child_index >= len(children[parent]):
      open_nodes.pop()
      # If we haven't omitted </LEFT> yet, do that first.
      if parent != -1 and (len(children[parent]) == 0 or children[parent][-1] < parent):
        out.append('</LEFT>')
      out.append('</RIGHT>')
    else:
      child = children[parent][child_index]
      word, _ = arcs[child]
      child_indices.append(child_index + 1)

      if child < parent and parent != -1:
        # Emit a word to the left
        out.append(word)
      else:
        # This child is to the right, but the previous one was to the left
        if parent != -1 and (prev_child == -1 or prev_child < parent):
          out.append('</LEFT>')
        # Emit a word to the right
        out.append(word)

      open_nodes.append(child)
      child_indices.append(0)
 
  return out

def parse_arcs(lines):
  arcs = []
  for line in lines:
    parts = line.split('\t')
    assert len(parts) == 8
    index = int(parts[0])
    word = parts[1]
    head = int(parts[6])
    assert len(arcs) == index - 1
    arcs.append((word, head - 1))
  return arcs

lines = []
for line in sys.stdin:
  line = line.strip()
  if not line:
    arcs = parse_arcs(lines)
    oracle = get_oracle(arcs)
    print ' '.join(oracle)
    lines = []
  else:
    lines.append(line)

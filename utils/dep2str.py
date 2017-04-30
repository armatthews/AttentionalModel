import sys

class Node:
  def __init__(self, word, parent):
    self.word = word
    self.left = []
    self.right = []
    self.done_with_left = parent == None
    self.parent = parent

  def add_word(self, word):
    if word == '</LEFT>':
      self.done_with_left = True
      return self
    elif word == '</RIGHT>':
      return self.parent

    node = Node(word, self)
    if self.done_with_left:
      self.right.append(node)
    else:
      self.left.append(node)
    return node

  def __str__(self):
    leftstr = ' '.join([str(node) for node in self.left])
    rightstr = ' '.join([str(node) for node in self.right])
    r = self.word
    if leftstr:
      r = leftstr + ' ' + r
    if rightstr:
      r = r + ' ' + rightstr
    return r

for line_num, line in enumerate(sys.stdin):
  words = line.split()
  if len(words) == 0:
    print
    continue

  lefts = len([word for word in words if word == '</LEFT>'])
  rights = len([word for word in words if word == '</RIGHT>'])
  real_words = len([word for word in words if word != '</LEFT>' and word != '</RIGHT>'])
  if lefts != real_words or lefts != rights - 1:
    print >>sys.stderr, 'Warning: Malformed dependency tree on line %d' % (line_num + 1)

  root = Node('<ROOT>', None)
  node = root
  for word in words:
    node = node.add_word(word)

  s = str(root)
  if s.startswith('<ROOT>'):
    s = s[len('<ROOT>'):].strip()
  #print real_words, lefts, rights, s
  print s

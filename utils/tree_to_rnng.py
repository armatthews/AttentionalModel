import re
import sys
import argparse

def find_next_paren(line, start):
  for i in range(start, len(line)):
    if line[i] == '(' or line[i] == ')':
      return i
  return None

def get_actions(line):
  i = 0
  actions = []
  while i < len(line):
    c = line[i]
    if c == ' ':
      i += 1
      continue
    if c == '(':
      next_paren = find_next_paren(line, i + 1)
      pos_and_term = line[i + 1 : next_paren]
      pos, term = pos_and_term.split(' ', 1)
      terms = term.split()
      if line[next_paren] == '(':
        actions.append('NT(%s)' % pos)
        for term in terms:
          actions.append('SHIFT(%s)' % term)
        i = next_paren
      else:
        if args.keep_preterms or len(terms) > 1:
          actions.append('NT(%s)'  % pos)
          for term in terms:
            actions.append('SHIFT(%s)' % term)
          actions.append('REDUCE')
        else:
          for term in terms:
            actions.append('SHIFT(%s)' % term)
        i = next_paren + 1
    elif c == ')':
      actions.append('REDUCE')
      i += 1
    else:
      next_paren = find_next_paren(line, i + 1)
      term = line[i:next_paren].strip()
      terms = term.split()
      for term in terms:
        actions.append('SHIFT(%s)' % term)
      i = next_paren
  return actions

parser = argparse.ArgumentParser()
parser.add_argument('--keep_preterms', '-k', action='store_true', help='Don\'t remove pre-terminals')
args = parser.parse_args()

for line in sys.stdin:
  line = line.strip()
  actions = get_actions(line)
  print ' '.join(actions)

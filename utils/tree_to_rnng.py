import re
import sys

def remove_unary(s): 
  while True:
    ns = re.sub(r'NT\(([^ ]*)\) SHIFT\(([^ ]*)\) REDUCE', r'SHIFT(\2)', s)
    if ns == s:
      return s
    s = ns

line_num = 0
for line in sys.stdin:
  line = line.strip()
  line_num += 1
  if not line:
    print
    continue

  i = 0
  parts = []
  while i < len(line):
    c = line[i]
    if c == '(':
      start = i + 1
      end = i + 1
      while line[end] != ' ':
        end += 1
      nt = line[start : end]
      parts.append('NT(%s)' % nt)
      i = end + 1
    elif c == ')':
      parts.append('REDUCE')
      i += 1
      while i < len(line) and line[i] == ' ':
        i += 1
    else:
      start = i
      end = i
      while end < len(line) and line[end] != ')':
        end += 1
      term = line[start : end]
      i = end
      while i < len(line) and line[i] == ' ':
        i += 1
      parts.append('SHIFT(%s)' % term)

  print remove_unary(' '.join(parts))

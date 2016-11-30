import sys

for line in sys.stdin:
  tokens = line.decode('utf-8').split()
  output = []
  for token in tokens:
    if token.startswith('NT('):
      assert token.endswith(')')
      output.append(' (%s' % token[3:-1])
    elif token.startswith('SHIFT('):
      assert token.endswith(')')
      output.append(' %s' % token[6:-1])
    elif token == 'REDUCE':
      output.append(')')
    else:
      print >>sys.stderr, 'Invalid token in RNNG input file: %s' % token.encode('utf-8')
      sys.exit(1)

  print ''.join(output).encode('utf-8').strip()

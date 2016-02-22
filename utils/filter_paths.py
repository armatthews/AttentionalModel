import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('vocab')
args = parser.parse_args()

vocab = set()
vocab.add('UNK')
vocab.add('</s>')
with open(args.vocab) as f:
  for line in f:
    vocab.add(line.strip())

for line in sys.stdin:
  cluster, word, count = line.split('\t')
  if word in vocab:
    sys.stdout.write(line)
    vocab.remove(word)

if len(vocab) > 0:
  print >>sys.stderr, 'Some words in vocab were not found in cluster file:'
  for i, word in enumerate(vocab):
    if i >= 10:
      print >>sys.stderr, '... and %d more' % (len(vocab) - 10)
      break
    print >>sys.stderr, word

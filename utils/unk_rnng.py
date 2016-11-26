import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('vocab')
parser.add_argument('-u', '--unk_token', required=False, default='UNK')
args = parser.parse_args()

vocab = set(map(lambda s: s.strip(), open(args.vocab).readlines()))

def replace_unk(word):
  if word.startswith('SHIFT'):
    assert word.endswith(')')
    w = word[6:-1]
    if w not in vocab:
      return 'SHIFT(%s)' % args.unk_token
  return word

for line in sys.stdin:
  words = line.strip().split()
  words = [replace_unk(word) for word in words]
  print ' '.join(words)

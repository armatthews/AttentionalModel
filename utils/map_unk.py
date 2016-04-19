import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('vocab')
parser.add_argument('-u', '--unk_token', required=False, default='UNK')
args = parser.parse_args()

vocab = set(map(lambda s: s.strip(), open(args.vocab).readlines()))

for line in sys.stdin:
  words = line.strip().split()
  words = [word if word in vocab else args.unk_token for word in words]
  print ' '.join(words)

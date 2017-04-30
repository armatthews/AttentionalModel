import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('vocab')
args = parser.parse_args()

vocab = set(map(lambda s: s.strip(), open(args.vocab).readlines()))

for line in sys.stdin:
  words = line.strip().split()
  for word in words:
    if word not in vocab:
      print word

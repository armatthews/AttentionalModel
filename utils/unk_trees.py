import sys
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('vocab')
parser.add_argument('-u', '--unk_token', required=False, default='UNK')
args = parser.parse_args()

vocab = set(map(lambda s: s.strip(), open(args.vocab).readlines()))

for line in sys.stdin:
  terminal_bits = re.findall(r' [^ ()]*[ )]', line)
  for terminal_bit in terminal_bits:
    word = terminal_bit[1:-1]
    if word not in vocab:
      while terminal_bit in line:
        i = line.find(terminal_bit)
        line = line[:i] + ' UNK' + terminal_bit[-1] + line[i + len(terminal_bit):]
      #line = re.sub(line, re.escape(terminal_bit), re.escape(' UNK' + terminal_bit[-1]))
  print line.strip()

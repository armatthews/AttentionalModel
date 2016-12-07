import sys
import math
import tempfile
import subprocess
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--rnng', default='/usr0/home/austinma/git/rnng/ntparse_0_2_32_64_16_60-pid35259.params', help='trained discriminative rnng parameters')
parser.add_argument('--model', required=True, help='trained attentional conditional rnng model')
parser.add_argument('--test_source', '-s', required=True, help='plain-text (tokenized) test set source')
parser.add_argument('--test_target', '-t', required=True, help='test set target in parenthesized tree format')
parser.add_argument('--test_target_raw', '-r', required=True, help='test set target in tokenized text format')
parser.add_argument('--train_target', default='/usr0/home/austinma/git/AttentionalModel/btec_easy/train.trees.en', help='rnng model\'s training trees in parenthesized tree format')
parser.add_argument('--rnng_train', default='/usr0/home/austinma/git/rnng/btec_easy/train', help='rnng model\'s training set')
parser.add_argument('--rnng_bracket_dev', default='/usr0/home/austinma/git/AttentionalModel/btec_easy/dev.trees.en', help='rnng model\'s bracked dev set')
parser.add_argument('--num_samples', '-n', type=int, default=100, help='number of samples')
args = parser.parse_args()

nt_parser_bin='/usr0/home/austinma/git/rnng/build/nt-parser/nt-parser'
get_oracle_bin='/usr0/home/austinma/git/rnng/get_oracle.py'
tree_to_rnng_bin='/usr0/home/austinma/git/AttentionalModel/utils/tree_to_rnng.py'
loss_bin='/usr0/home/austinma/git/AttentionalModel/bin/loss'

output = subprocess.check_output(['python', get_oracle_bin, args.train_target, args.test_target])
_, test_oracle = tempfile.mkstemp()
with open(test_oracle, 'w') as f:
  f.write(output)
  f.close()

output = subprocess.check_output([nt_parser_bin, '-m', args.rnng, '-T', args.rnng_train, '-p', test_oracle, '-s', str(args.num_samples), '-x', '-C', args.rnng_bracket_dev])

samples = defaultdict(list)

for line in output.split('\n'):
  if not line.strip():
    continue
  sent_id, prob, sample = line.split('|||')
  sent_id = int(sent_id.strip())
  prob = float(prob.strip())
  sample = sample.strip()
  samples[sent_id].append((prob, sample))
  #print >>sys.stderr, '%d ||| %s ||| %f' % (sent_id, sample, prob)

for k in samples:
  assert len(samples[k]) == args.num_samples

_, test_src = tempfile.mkstemp()
_, test_tgt = tempfile.mkstemp()
process = subprocess.Popen(['python', tree_to_rnng_bin, '-k'], stdin=subprocess.PIPE, stdout=open(test_tgt, 'w'))
with open(test_src, 'w') as f:
  for i, src in enumerate(open(args.test_source)):
    src = src.strip()
    for rnng_prob, sample in samples[i]:
      f.write(src + '\n')
      process.stdin.write(sample + '\n')
process.stdin.close()
process.wait()

with open(test_tgt, 'r') as f:
  for i, src in enumerate(open(args.test_source)):
    src = src.strip()
    for rnng_prob, sample in samples[i]:
      tgt = f.readline().strip()
      #print >>sys.stderr, '%s ||| %s ||| %f' % (src, tgt, rnng_prob)

output = subprocess.check_output([loss_bin, '--dynet_mem', '4000', '--model', args.model, '--input_source', test_src, '--input_target', test_tgt]).split('\n')
output_line_num = 0
log_probs = []
for i, src in enumerate(open(args.test_source)):
  s = 0.0
  for rnng_prob, sample in samples[i]:
    model_prob = -float(output[output_line_num].split('|||')[1].strip())
    s += math.exp(model_prob - rnng_prob)
    output_line_num += 1
  s /= 100
  log_probs.append(math.log(s))

total_words = 0
for i, tgt in enumerate(open(args.test_target_raw)):
  tgt = tgt.split()
  total_words += len(tgt)
  #print '%d ||| %f' % (i, -log_probs[i] / len(tgt))
  print '%d ||| %f' % (i, -log_probs[i])

xent = -sum(log_probs) / total_words
perp = math.exp(xent)
print 'Total: %f xent (%f perp) over %d words' % (xent, perp, total_words)

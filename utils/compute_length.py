import re
import sys
import argparse
from math import exp
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('kbest')
parser.add_argument('refs')
parser.add_argument('--wrap_refs', action='store_true')
args = parser.parse_args()

refs = []
with open(args.refs) as ref_file:
	for line in ref_file:
		line = line.strip()
		if args.wrap_refs:
			line = '<s> ' + line + ' </s>'
		ref = tuple(line.split())
		refs.append(ref)
warned = False
for line in open(args.kbest):
	parts = line.split('|||')
	sent_id = int(parts[0])
	hyp = tuple(parts[1].strip().split())
	hyp = tuple(word for word in hyp if re.match(r'\|[0-9]*-[0-9]*\|', word) is None)
	ref = refs[sent_id]
	if not warned and len(hyp) > 0 and len(ref) > 0:
		if hyp[0] == '<s>' and ref[0] != '<s>':
			warned = True
			print >>sys.stderr, 'WARNING: Hypotheses use <s> but refs do not!'
		elif hyp[0] != '<s>' and ref[0] == '<s>':
			warned = True
			print >>sys.stderr, 'WARNING: References use <s> but hyps do not!'

	score = 100 * len(hyp) / len(ref)
	if score > 0.01:
		print line.strip(), '|||', score

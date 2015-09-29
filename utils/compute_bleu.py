import re
import sys
import argparse
from math import exp
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('kbest')
parser.add_argument('refs')
parser.add_argument('--wrap_refs', action='store_true')
parser.add_argument('--plus', type=float, default=1.0, help='Amount to smooth BLEU score by. Default is 1.0 = BLEU+1')
args = parser.parse_args()

refs = []
with open(args.refs) as ref_file:
	for line in ref_file:
		line = line.strip()
		if args.wrap_refs:
			line = '<s> ' + line + ' </s>'
		ref = tuple(line.split())
		refs.append(ref)
#assert len(refs) == 1370

def find_ngrams(hyp, n):
	ngrams = Counter()
	for i in range(0, len(hyp) - n + 1):
		j = i + n
		ngrams[hyp[i:j]] += 1
	return ngrams

N = 4
ref_ngram_cache = {}
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

	p = [0 for n in range(N)]
	for n in range(0, N):
		hyp_ngrams = find_ngrams(hyp, n + 1)
		if (sent_id, n) not in ref_ngram_cache:
			ref_ngrams = find_ngrams(ref, n + 1)
			ref_ngram_cache[sent_id, n] = ref_ngrams
		else:
			ref_ngrams = ref_ngram_cache[sent_id, n]

		matches = hyp_ngrams & ref_ngrams
		p[n] = (sum(matches.values()) + args.plus) / (sum(hyp_ngrams.values()) + args.plus)

	bp = exp(1.0 - len(ref) / len(hyp)) if len(hyp) < len(ref) else 1.0
	score = reduce(lambda x, y: x*y, p) ** (1.0 / N) * bp
	print line.strip(), '|||', score

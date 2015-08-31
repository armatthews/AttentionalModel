import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--one_best', action='store_true')
args = parser.parse_args()

scores = {}
for line in sys.stdin:
	parts = [part.strip() for part in line.split('|||')]
	sent_id = parts[0]
	score = float(parts[-1])
	if sent_id not in scores:
		scores[sent_id] = score
	elif score > scores[sent_id] and not args.one_best:
		scores[sent_id] = score

print >>sys.stderr, 'Read %d sentences.' % (len(scores))
print sum(scores.values()) / len(scores)

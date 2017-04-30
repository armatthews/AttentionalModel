import sys
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser('Map each word of a corpus to its corresponding cluster')
parser.add_argument('cluster_map', help='Tab delimited file of the format cluster[TAB]word')
args = parser.parse_args()

cluster_map = defaultdict(lambda: '<unk>')
with open(args.cluster_map) as f:
	for line in f:
		parts = line.decode('utf-8').strip().split('\t')
		if len(parts) < 2:
			print >>sys.stderr, 'Invalid line in cluster map: %s' % line.strip()
			sys.exit(1)
		cluster, word = parts[:2]
		cluster_map[word] = cluster

for line in sys.stdin:
	line = line.decode('utf-8').strip()
	words = line.split()
	clusters = [cluster_map[word] for word in words]
	print ' '.join(clusters).encode('utf-8')

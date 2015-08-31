# This script is useful if you have parallalized decoding of a test set.
# That can cause the k-best list indices to be non-unique.
# For example if you divide a 10-sentence test set over three cores your
# sentence ids might go: 0 1 2 3 0 1 2 0 1 2.
# This script fixes that problem and makes it go: 0 1 2 3 4 5 6 7 8 9
sed 's/ ||| /\t/g' | awk -F $'\t' 'BEGIN {i = 0; prev= 0} {if ($1 != prev) {prev = $1; i += 1; } printf i; for (j=2; j <= NF; ++j) { printf FS $j } print ""}' | sed 's/\t/ ||| /g'

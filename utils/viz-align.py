import sys
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')
parser.add_argument('alignments')
parser.add_argument('gradients')
args = parser.parse_args()

source = [['<s>'] + line.split() + ['</s>'] for line in open(args.source).readlines()]
target = [line.split() + ['</s>'] for line in open(args.target).readlines()]

assert len(source) == len(target)
for i in range(len(source)):
  source[i] = [word.replace('&', '&amp;').replace('<', '&lt;') for word in source[i]]
  target[i] = [word.replace('&', '&amp;').replace('<', '&lt;') for word in target[i]]

word_num = 0
sent_num = 0

def val_to_char(f):
  if f < 0.25: return ' '
  if f < 0.5: return '.'
  if f < 0.75: return 'x'
  return '@'

def grad_to_char(f):
  assert f >= -8
  assert f <= 8
  if f < -2.5: return '&darr;&darr;&darr;'
  if f < -0.5: return '&darr;&darr;'
  if f < -0.1: return '&darr;'
  if f > 0.1: return '&uarr;'
  if f > 0.5: return '&uarr;&uarr;'
  if f > 2.5: return '&uarr;&uarr;&uarr;'
  return ''

def val_to_num(f):
  return int((1.0 - f) * 255 + 0.5)

def val_to_color(f):
  n = val_to_num(f)
  return '%02x%02x%02x' % (n, n ,n)

def grad_to_color(g):
  assert g >= 0.0
  if g > 5.0:
    g = 5.0
  n = int(g / 5.0 * 255 + 0.5)
  return '%02x0000' % (n)

def begin_document():
  print '<html>'
  print '<head>'
  print '<meta charset="UTF-8">'
  print '<style>'
  print 'table { border-spacing: 0; }'
  for i in range(0, 256):
    print '.cell%02x,' % i,
  print '.cell { width: 30px; border: 1px solid black; text-align: center;}'
  for i in range(0, 256):
    print '.cell%02x { background-color: %02x%02x%02x; }' % (i, i, i, i)
  print '.vert { writing-mode:vertical-lr; }'
  print '.source { width: 30px; text-align: center; }'
  print '</style>'
  print '</head>'
  print '<body>'

def end_document():
  print '</body>'
  print '</html>'

def begin_sentence():
  print '<table>'
  print '<tr>'
  print '<td></td>'
  for word in source[sent_num]:
    print '<td class="source"><span class="vert">%s</span></td>' % word
  print '<td></td>'
  print '</tr>'

def end_sentence():
  print '<tr>'
  print '<td></td>'
  for word in source[sent_num]:
    print '<td class="source"><span class="vert">%s</span></td>' % word
  print '<td></td>'
  print '</tr>'
  print '</table>'
  print '<br /><br />'
  #print ' '.join(source[sent_num])
  print

def emit_row(alignments, gradients):
  assert len(alignments) == len(source[sent_num])
  assert len(gradients) == len(source[sent_num])
  grad_len = math.sqrt(sum(g**2 for g in gradients))
  grad_color = grad_to_color(grad_len)
  target_word = target[sent_num][word_num]

  print '<tr>'
  print '<td style="text-align: right; white-space: nowrap; color: %s;">%s</td>' % (grad_color, target_word)
  for a, g in zip(alignments, gradients):
    n = val_to_num(a)
    print '<td class="cell%02x">%s</td>' % (n, grad_to_char(g))
  print '<td style="white-space: nowrap; color:%s;">%s</td>' % (grad_color, target_word)
  print '</tr>'

begin_document()
begin_sentence()
in_sentence = True
for aline, gline in zip(open(args.alignments), open(args.gradients)):
  aline = aline.strip()
  gline = gline.strip()
  if aline == '':
    assert gline == ''
    assert word_num == len(target[sent_num])
    end_sentence()
    in_sentence = False
    sent_num += 1
    word_num = 0
    continue

  if not in_sentence:
    begin_sentence()
    in_sentence = True
  av = map(float, aline.split())
  gv = map(float, gline.split())
  emit_row(av, gv)
  word_num += 1

if in_sentence:
  end_sentence()
end_document()

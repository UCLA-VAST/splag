#!/usr/bin/python3
import sys

import scipy.stats


def main():
  ratio = []
  for line in sys.stdin:
    if '#edges visited:' in line:
      ratio.append(float(line.split()[7][1:-1]))
  print(scipy.stats.hmean(ratio))


if __name__ == '__main__':
  main()

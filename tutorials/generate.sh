#!/bin/sh

for f in $(ls *.ipynb); do
  for fmt in markdown html; do
      python -m jupyter nbconvert --to=${fmt} $f
  done
done

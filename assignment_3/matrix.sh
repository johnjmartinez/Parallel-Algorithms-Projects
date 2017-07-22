#!/bin/bash

rows=$1
cols=$2

out=matrix_${rows}_${cols}.txt

python ./createMatrix.py $rows $cols
echo "$rows $cols" > $out
cat tmp >> $out
rm tmp


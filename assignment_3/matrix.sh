#!/bin/bash

rows=$1
cols=$2

python ./createMatrix.py $rows $cols
echo "$rows $cols" > input_${rows}_${cols}.txt
cat tmp >> input_${rows}_${cols}.txt
rm tmp


#!/bin/bash

rows=$1


out=vector_${rows}.txt

python ./createVector.py $rows
#echo "$rows" > $out
cat tmp >> $out
rm tmp

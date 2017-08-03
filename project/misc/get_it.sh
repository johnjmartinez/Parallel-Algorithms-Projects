#!/bin/bash
g=$1

algs="quickSort2 brickSort2 brickSort3 thrustSortSimple thrustOptmSort cubDevRadixSort_ mgpuMergeSort"  
d="cubDevRadixSort2"
b="cubBlkRadixSort"

rm $g; touch $g
for i in `seq 15 25`; do 
    n=$((1<<i))
    echo -e "$n \n" > tmp
    for a in $algs; do 
        t=`grep -A4 $a $g.log.$n | grep Thr | perl -pe "s/.+put =\s+(\S+) MEl.+/\1/"`
        echo -e "$t\t"  >> tmp
    done
    paste $g tmp > x
    mv x $g
done 

rm y; touch y
for i in `seq 15 25`; do 
    n=$((1<<i))
    echo -e "\n$n \n" > tmp
    grep -A6 $d $g.log.$n | grep Thr | perl -pe "s/.+put =\s+(\S+) MEl.+/\1/" >> tmp
    echo >> tmp
    grep -A4 $b $g.log.$n | grep Thr | perl -pe "s/.+put =\s+(\S+) MEl.+/\1/" >> tmp
    paste y tmp > x
    mv x y 
done 

cat y >> $g
rm y

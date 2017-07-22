#!/bin/bash
set -x

ignore="-Wno-deprecated-gpu-targets"

## COMPILE 
#Q1
nvcc array_minimum_par.cu -o q1a $ignore
nvcc array_output.cu -o q1b $ignore
#Q2
g++  -std=c++11 rangeCountSequential.cpp -o r_seq $ignore
nvcc -std=c++11 rangeCount.cu -o r_par $ignore
#Q3
gcc radixSortSequential.c -o q3_seq $ignore
nvcc radix_sort.cu -o q3_par $ignore

### RUN
for x in inp[1-5].txt
do
    num=$(head -n1 $x)
    tail -n1 $x  > inp_${num}.txt  
    echo inp_${num}.txt
    echo "Q1"    
    ./q1a inp_${num}.txt $num
    ./q1b inp_${num}.txt $num
    echo "Q2"    
    ./r_seq  inp_${num}.txt  $num
    ./r_par  inp_${num}.txt  $num
    echo "Q3"    
    ./q3_seq  inp_${num}.txt  $num
    ./q3_par  inp_${num}.txt  $num    
    echo
done


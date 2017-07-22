#!/bin/bash
ignore="-Wno-deprecated-gpu-targets"

### COMPILE 

#Q2
g++  -std=c++11 rangeCountSequential.cpp -o q2_seq $ignore
nvcc -std=c++11 rangeCount.cu -o q2_par $ignore

for x in inp[1-5].txt
do
    num=$(head -n1 $x)
    blks=$(( $num/1024 + 1 ))
    tail -n1 $x  > inp_${num}.txt  
    echo inp_${num}.txt
    
    for cu in array_*.cu radix_sort.cu
    do
        perl -pe "s/(define nthreads).*/\1 ${num}/; s/(define nblocks).*/\1 ${blks}/" $cu > tmp
        mv tmp $cu
    done
    
    #Q1
    nvcc array_minimum_par.cu -o q1a $ignore
    nvcc array_output.cu -o q1b $ignore
    
    #Q3
    nvcc radix_sort.cu -o q3_par $ignore

### RUN

    echo "--- Q1" ; set -x 
    ./q1a inp_${num}.txt $num
    ./q1b inp_${num}.txt $num
    set +x ; echo "--- Q2" ; set -x 
    ./q2_seq  inp_${num}.txt  $num
    ./q2_par  inp_${num}.txt  $num
    set +x ; echo "--- Q3" ; set -x 
    ./q3_par  inp_${num}.txt  $num    
    set +x ; echo "####################################################################"
    echo
done


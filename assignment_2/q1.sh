#!/bin/bash

### COMPILE 
for x in inp[1-5].txt
do
    num=$(head -n1 $x)
    blks=$(( $num/1024 + 1 ))
    tail -n1 $x  > inp_${num}.txt  
    echo inp_${num}.txt
    
    for cu in array_*.cu
    do
        perl -pe "s/(define nthreads).*/\1 ${num}/; s/(define nblocks).*/\1 ${blks}/" $cu > tmp
        mv tmp $cu
    done
    
    #Q1
    nvcc array_minimum_par.cu -o q1a $ignore
    nvcc array_output.cu -o q1b $ignore
    
### RUN

    echo "--- Q1" ; set -x 
    ./q1a inp_${num}.txt $num
    ./q1b inp_${num}.txt $num
    set +x ; echo "####################################################################"
    echo
done


#!/bin/bash
set -x

python createInput.py $((1024*1024)) # 1M   

g++ sortSequential.cpp -std=c++14 -o sortSequential
time ./sortSequential input.txt     # creates output_serial.txt

mpic++ hyperQuickSortMpi.cpp -std=c++14 -o sortParallel
time mpirun -n 16  sortParallel input.txt   # creates output.txt 

diff output_serial.txt output.txt   # check

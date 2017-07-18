#!/bin/bash
set -x

python createInput.py 10000

g++ sortSequential.cpp -std=c++14 -o sortSequential
./sortSequential input.txt

mpic++ hyperQuickSortMpi.cpp -std=c++14 -o sortParallel
mpirun -n 4  sortParallel input.txt

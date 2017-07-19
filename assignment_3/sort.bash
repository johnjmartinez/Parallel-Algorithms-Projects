#!/bin/bash
set -x

python createInput.py 48

g++ sortSequential.cpp -std=c++14 -o sortSequential
./sortSequential input.txt

mpic++ hyperQuickSortMpi.cpp -std=c++14 -o sortParallel
mpirun -n 8  sortParallel input.txt

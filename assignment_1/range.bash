#!/bin/bash
set -x

g++  -std=c++11 rangeCountSequential.cpp -o r_seq
nvcc -std=c++11 rangeCount.cu -o r_par

set +x

echo
echo "Expected results using file_450.txt "
echo "(first line Q2a,b, second line Q2c) : "
echo " 50  49  40  36  48  50  36  52  51  38 "
echo " 50  99  139 175 223 273 309 361 412 450"
echo
echo "Sequential run:"
set -x
./r_seq  file_450.txt  450
set +x
echo
echo "Parallel run:"
set -x
./r_par  file_450.txt  450
set +x
echo

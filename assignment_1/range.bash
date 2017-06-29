#!/bin/bash
set -x

g++  -std=c++11 rangeCountSequential.cpp -o r_seq
nvcc -std=c++11 rangeCount.cu -o r_par

set +x

echo
echo "Expected results using file_1000000.txt "
echo "(first line Q2a,b, second line Q2c) : "
echo " 100128	100146	99786	99946	100170	99652	100130	100182	99926	99934	"
echo " 100128	200274	300060	400006	500176	599828	699958	800140	900066	1000000	"
echo
echo "Sequential run:"
set -x
./r_seq  file_1000000.txt  1000000
set +x
echo
echo "Parallel run:"
set -x
./r_par  file_1000000.txt  1000000
set +x
echo

#!/bin/bash
set -x

nvcc array_minimum_par.cu -o q1a
nvcc array_output.cu -o q1b

./q1a  file_450.txt 450 # OK
echo
echo
./q1b  file_450.txt 450 # OK

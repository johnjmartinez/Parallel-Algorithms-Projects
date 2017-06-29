#!/bin/bash
set -x

nvcc array_minimum_par.cu -o q1a
nvcc array_output.cu -o q1b

./q1a  file_500.txt 500 # OK
./q1b  file_500.txt 500 # OK

./q1a  file_750.txt 750 # segmentation fault
./q1b  file_750.txt 750 # core dump

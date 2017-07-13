#!/bin/bash
set -x

g++ sortSequential.cpp -std=c++17 -o sortSequential
python createInput.py 10000
./sortSequential input.txt

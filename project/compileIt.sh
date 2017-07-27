#!/bin/bash

module load cuda
module load cxx11

set -x
cuda=$1
time nvcc $cuda -arch=compute_35 -code=sm_35 -I cuda_common/inc/ \
    -o ${cuda%.cu}

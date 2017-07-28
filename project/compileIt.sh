#!/bin/bash

module load cuda
module load cxx11

set -x
cuda=$1
time nvcc $cuda -arch=compute_30 -code=sm_30 -I cuda_common/inc/ \
    -o ${cuda%.cu}

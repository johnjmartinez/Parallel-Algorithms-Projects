#!/bin/bash

module load cuda
module load cxx11

set -x
_cu_=$1
time nvcc $_cu_ \
    -I cuda_common/inc/ \
    -I cub/ \
    -l cuda \
    -std=c++11 \
    -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES \
    -o ${_cu_%.cu}_

# -arch=compute_30 -code=sm_30 \ # breaking cub

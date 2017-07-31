#!/bin/bash

#module load cuda
#module load cxx11

set -x
_cu_=$1
time nvcc $_cu_ -O3 \
    -I cuda_common/inc/ \
    -I cub/ \
    -std=c++11 \
    -l cuda \
    -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES \
    -arch=compute_30 -code=sm_30 \
    -o ${_cu_%.cu}_

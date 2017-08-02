#!/bin/bash
# for x in *.cu; do ./compileIt.sh $x; done

#module load cuda
#module load cxx11

set -x
_cu_=$1
nvcc $_cu_ -O3 \
    -I cuda_common/inc/ -I cub/ -I ./\
    -lineinfo -l cuda -use_fast_math \
    -std=c++11 --expt-extended-lambda \
    -arch=compute_61 -code=sm_61 \
    -o ${_cu_%.cu}_ 

#    -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES \

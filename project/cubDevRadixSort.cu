// time nvcc $cu -I ../cuda_common/inc/  -I cub/ -l cuda -std=c++11 -o ${cu%.cu} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES

#include <stdio.h>
#include <vector>
#include <functional>
#include <stdint.h>
#include <cuda.h>

#include <cub.cuh>
#include <cuda_common.h>    

// Parameters
//const int defaultDATASIZE = 8<<20; 
double MIN_BENCH_TIME = 1.0;  // mimimum seconds to run each bechmark

template <typename T>
__global__ void randFill (T *d_array, uint32_t size) {

    const uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= size)  return;

    uint32_t rnd = idx*1234567891u;

    rnd = 29943829*rnd + 1013904223;    // https://en.wikipedia.org/wiki/Linear_congruential_generator
    rnd = 29943829*rnd + 1013904223;

    uint64_t rnd1 = rnd;

    rnd = 29943829*rnd + 1013904223;
    rnd = 29943829*rnd + 1013904223;

    d_array[idx] = T(rnd1<<32) + rnd;
}

template <typename elem>
double devRadixSort (int SORT_BYTES, size_t n, void *d_array0, cudaEvent_t &start, cudaEvent_t &stop) {
    
    int begin_bit = 0,  end_bit = SORT_BYTES*8; // Bit subrange [begin_bit, end_bit) of differentiating elem bits
    auto d_array = (elem*) d_array0;

    cub::DoubleBuffer<elem> d_elems (d_array, d_array + n); // Create DoubleBuffer to wrap pair of Dev pointers

    // Determine temp Dev storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    checkCudaErrors( cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_elems, n, begin_bit, end_bit));

    checkCudaErrors( cudaMalloc (&d_temp_storage, temp_storage_bytes)); // Allocate temp storage

    int numIterations = 0;
    double totalTime = 0;

    for ( ; totalTime < MIN_BENCH_TIME; numIterations++) {
       
        randFill<elem> <<< n/1024+1, 1024 >>> (d_array, n);  // Fill source buffer with random numbers
        checkCudaErrors( cudaDeviceSynchronize());

        checkCudaErrors( cudaEventRecord (start, nullptr));

        // Run sorting operation
        checkCudaErrors( cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_elems, n, begin_bit, end_bit) );
        
        checkCudaErrors( cudaEventRecord (stop, nullptr)); // Record time
        checkCudaErrors( cudaDeviceSynchronize());
        
        float time;
        checkCudaErrors( cudaEventElapsedTime (&time, start, stop));
        totalTime += time/1000; // converts milliseconds to seconds
    }

    checkCudaErrors( cudaFree (d_temp_storage)); // Release temp storage
    return totalTime/numIterations;
}

int main (int argc, char **argv) {

    const int DATASIZE = atoi(argv[1]); // defaultDATASIZE;

    DisplayCudaDevice();

    void* d_array;
    checkCudaErrors( cudaMalloc(&d_array, 4*DATASIZE*sizeof(int)));

    cudaEvent_t start, stop;
    checkCudaErrors( cudaEventCreate(&start));
    checkCudaErrors( cudaEventCreate(&stop));

    auto print = [&] (int bytes, int elemsize, double totalTime) {
        printf("%d/%d: Throughput =%9.3lf MElements/s, Time = %.3lf ms\n",
               bytes, elemsize, 1e-6 * DATASIZE / totalTime, totalTime*1000);
    };

    printf("Sorting %d elements:\n", DATASIZE);
    for(int i=1; i<=4; i++)  
        print (i, 4, devRadixSort<int>(i, DATASIZE, d_array, start, stop));  
    printf("\n");
    return 0;
}

// time nvcc $cu -I ../cuda_common/inc/  -I cub/ -l cuda -std=c++11 -o ${cu%.cu} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES

#include <stdio.h>
#include <vector>
#include <functional>
#include <stdint.h>

#include <helper_functions.h>   // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>        // helper functions for CUDA error checking and initialization
#include <cuda.h>

#include <cub.cuh>
#include <cuda_common.h>       // cub cuda-specific helper functions
#include <wall_clock_timer.h>  // cub StartTimer() and GetTimer()
#include <cpu_common.h>        // cub helper functions

// Parameters
const int defaultNumElements = 32<<20; // 32M
double MIN_BENCH_TIME = 0.5;  // mimimum seconds to run each bechmark

template <typename T>
__global__ void fill_with_random (T *d_array, uint32_t size) {

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

template <typename Key>
double key_sort (int SORT_BYTES, size_t n, void *d_array0, cudaEvent_t &start, cudaEvent_t &stop) {
    
    int begin_bit = 0,  end_bit = SORT_BYTES*8; // Bit subrange [begin_bit, end_bit) of differentiating key bits
    auto d_array = (Key*) d_array0;

    cub::DoubleBuffer<Key> d_keys (d_array, d_array + n); // Create DoubleBuffer to wrap pair of Dev pointers

    // Determine temp Dev storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    checkCudaErrors( cub::DeviceRadixSort::SortKeys (d_temp_storage, temp_storage_bytes, d_keys, n, begin_bit, end_bit));

    checkCudaErrors( cudaMalloc (&d_temp_storage, temp_storage_bytes)); // Allocate temp storage

    int numIterations = 0;
    double totalTime = 0;

    for ( ; totalTime < MIN_BENCH_TIME; numIterations++) {
       
        fill_with_random<Key> <<< n/1024+1, 1024 >>> (d_array, n);  // Fill source buffer with random numbers
        checkCudaErrors( cudaDeviceSynchronize());

        checkCudaErrors( cudaEventRecord (start, nullptr));

        // Run sorting operation
        checkCudaErrors( cub::DeviceRadixSort::SortKeys (d_temp_storage, temp_storage_bytes, d_keys, n, begin_bit, end_bit));
        
        checkCudaErrors( cudaEventRecord (stop, nullptr)); // Record time
        checkCudaErrors( cudaDeviceSynchronize());
        float start_stop;
        checkCudaErrors( cudaEventElapsedTime (&start_stop, start, stop));
        totalTime += start_stop/1000; // converts milliseconds to seconds
    }

    checkCudaErrors( cudaFree (d_temp_storage)); // Release temp storage
    return totalTime/numIterations;
}

int main (int argc, char **argv) {
    bool full = false;
    int numElements = defaultNumElements;

    while (*++argv) {
      ParseBool (*argv, "full", "", &full) ||
      ParseInt  (*argv, "",         &numElements) ||
      (printf ("radix_sort: benchmark CUB Radix Sort with various parameters.\n"
               "Usage: radix_sort [N]\n"
               " - N is number [of millions] of elements to test\n"
              ),
       exit(1), 1);
    }

    if (numElements < 16384) numElements <<= 20;

    DisplayCudaDevice();

    void* d_array;
    checkCudaErrors( cudaMalloc(&d_array, 4*numElements*sizeof(uint64_t)));

    cudaEvent_t start, stop;
    checkCudaErrors( cudaEventCreate(&start));
    checkCudaErrors( cudaEventCreate(&stop));

    auto print = [&] (int bytes, int keysize, int valsize, double totalTime) {
        char valsize_str[100];
        sprintf(valsize_str, (valsize? "+%d": "  "), valsize);
        printf("%d/%d%s: Throughput =%9.3lf MElements/s, Time = %.3lf ms\n",
               bytes, keysize, valsize_str, 1e-6 * numElements / totalTime, totalTime*1000);
    };

    printf("Sorting %dM elements:\n", numElements>>20);
    {for(int i=1;i<=4;i++)  print (i, 4, 0, key_sort <uint32_t> (i, numElements, d_array, start, stop));  printf("\n");}
    {for(int i=1;i<=8;i++)  print (i, 8, 0, key_sort <uint64_t> (i, numElements, d_array, start, stop));  printf("\n");}
    return 0;
}

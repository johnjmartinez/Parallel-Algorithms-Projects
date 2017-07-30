// COMPILE WITH:  nvcc thrustRadixSort.cu -I cuda_common/inc/ -o thrustRadixSort

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

#include <helper_cuda.h>
#include <cuda_common.h> 

#include <algorithm>
#include <time.h>
#include <limits.h>

template <typename T>
bool testSort(int argc, char **argv) {

    int keybits = 32; // INT32
    unsigned int DATASIZE = (8<<20);
    unsigned int numIterations = 20;
    
    printf("Sorting %d %d-bit int elems\n", DATASIZE, keybits);

    int deviceID = -1;
    if (cudaSuccess == cudaGetDevice(&deviceID)) {
        cudaDeviceProp devprop;
        cudaGetDeviceProperties(&devprop, deviceID);
        unsigned int totalMem = 2 * DATASIZE * sizeof(T);

        if (devprop.totalGlobalMem < totalMem) {
            printf("Error: insufficient amount of memory to sort %d elements.\n", DATASIZE);
            printf("%d bytes needed, %d bytes available\n", (int) totalMem, (int) devprop.totalGlobalMem);
            exit(EXIT_SUCCESS);
        }
    }

    // Create host vectors
    thrust::host_vector<T> h_elems(DATASIZE);
    thrust::host_vector<T> h_elemsSorted(DATASIZE);

    // Fill up with some random data
    thrust::default_random_engine rng(clock());
    thrust::uniform_int_distribution<int> u(0, INT_MAX);
    for (int i = 0; i < (int)DATASIZE; i++) h_elems[i] = u(rng);

    // Create GPU device vectors
    thrust::device_vector<T> d_elems;

    // run multiple iterations to compute an average sort time
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    float totalTime = 0;
    float time = 0;

    for (unsigned int i = 0; i < numIterations; i++) {
        
        checkCudaErrors(cudaEventRecord(start_event, nullptr));

        d_elems = h_elems; // reset data before sort (copy from host to device)
        thrust::sort(d_elems.begin(), d_elems.end());

        checkCudaErrors( cudaEventRecord(stop_event, nullptr) );
        checkCudaErrors( cudaEventSynchronize(stop_event) );

        time = 0;
        checkCudaErrors(cudaEventElapsedTime(&time, start_event, stop_event));
        totalTime += time;
    }

    time = totalTime / numIterations / 1000; // in secs
    printf("Throughput =%9.3lf MElements/s, Time = %.3lf ms\n", 1e-6 * DATASIZE / time, time*1000);
    getLastCudaError("after radixsort");

    // Get results back to host for correctness checking
    thrust::copy(d_elems.begin(), d_elems.end(), h_elemsSorted.begin());
    getLastCudaError("copying results to host memory");

    // Check results
    bool sortResult = thrust::is_sorted(h_elemsSorted.begin(), h_elemsSorted.end());

    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));

    return sortResult;
}

int main(int argc, char **argv) {
    
    DisplayCudaDevice();

    bool sortResult = testSort<int>(argc, argv);
    
    checkCudaErrors( cudaDeviceReset() );
    printf(sortResult ? "Test passed\n" : "Test failed!\n");
}

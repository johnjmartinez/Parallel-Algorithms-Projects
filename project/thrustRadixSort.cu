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

#include <algorithm>
#include <time.h>
#include <limits.h>

template <typename T, bool floatKeys>
bool testSort(int argc, char **argv) {
    int cmdVal;
    int keybits = 32; // UINT32

    unsigned int numElements = (32<<20);
    bool quiet = (checkCmdLineFlag(argc, (const char **)argv, "quiet") == true);

    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "n");
        numElements = cmdVal;

        if (cmdVal < 0) {
            printf("Error: elements must be > 0, elements=%d is invalid\n", cmdVal);
            exit(EXIT_SUCCESS);
        }
    }

    unsigned int numIterations = (numElements >= 16777216) ? 20 : 100;

    if (checkCmdLineFlag(argc, (const char **)argv, "iterations")) {
        cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
        numIterations = cmdVal;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
        printf("Command line:\nradixSortThrust [-option]\n");
        printf("Valid options:\n");
        printf("-n=<N>        : number of elements to sort\n");
        printf("-quiet        : Output only the number of elements and the time to sort\n");
        printf("-help         : Output a help message\n");
        exit(EXIT_SUCCESS);
    }

    if (!quiet)
        printf("Sorting %d %d-bit unsigned int keys only\n", numElements, keybits);

    int deviceID = -1;

    if (cudaSuccess == cudaGetDevice(&deviceID)) {
        cudaDeviceProp devprop;
        cudaGetDeviceProperties(&devprop, deviceID);
        unsigned int totalMem = 2 * numElements * sizeof(T);

        if (devprop.totalGlobalMem < totalMem) {
            printf("Error: insufficient amount of memory to sort %d elements.\n", numElements);
            printf("%d bytes needed, %d bytes available\n", (int) totalMem, (int) devprop.totalGlobalMem);
            exit(EXIT_SUCCESS);
        }
    }

    // Create host vectors
    thrust::host_vector<T> h_keys(numElements);
    thrust::host_vector<T> h_keysSorted(numElements);
    thrust::host_vector<unsigned int> h_values;

    // Fill up with some random data
    thrust::default_random_engine rng(clock());
    thrust::uniform_int_distribution<unsigned int> u(0, UINT_MAX);
    for (int i = 0; i < (int)numElements; i++)  h_keys[i] = u(rng);

    // Create GPU device vectors
    thrust::device_vector<T> d_keys;
    thrust::device_vector<unsigned int> d_values;

    // run multiple iterations to compute an average sort time
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    float totalTime = 0;

    for (unsigned int i = 0; i < numIterations; i++) {
        
        d_keys= h_keys; // reset data before sort (copy from host to device)

        checkCudaErrors(cudaEventRecord(start_event, 0));

        thrust::sort(d_keys.begin(), d_keys.end());

        checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));

        float time = 0;
        checkCudaErrors(cudaEventElapsedTime(&time, start_event, stop_event));
        totalTime += time;
    }

    totalTime /= (1.0e3f * numIterations);
    printf("radixSort, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements\n",
           1.0e-6f * numElements / totalTime, totalTime, numElements);

    getLastCudaError("after radixsort");

    // Get results back to host for correctness checking
    thrust::copy(d_keys.begin(), d_keys.end(), h_keysSorted.begin());
    getLastCudaError("copying results to host memory");

    // Check results
    bool bTestResult = thrust::is_sorted(h_keysSorted.begin(), h_keysSorted.end());

    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));

    if (!bTestResult  && !quiet) return false;
    return bTestResult;
}

int main(int argc, char **argv) {
    
    printf("%s Starting...\n\n", argv[0]); // Start logs

    findCudaDevice(argc, (const char **)argv); // cuda_common/inc/helper_cuda.h

    bool bTestResult = false;
    bTestResult = testSort<unsigned int, false>(argc, argv);

    checkCudaErrors(cudaDeviceReset());
		
    printf(bTestResult ? "Test passed\n" : "Test failed!\n");
}

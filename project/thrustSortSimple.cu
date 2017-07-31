// COMPILE: nvcc thrustSortSimple.cu -I cuda_common/inc/ -o thrustSort

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cuda_common.h>     

static void cuda_assert(const cudaError_t code, const char* const file, const int line, const bool abort) {
    if (code != cudaSuccess) {
        fprintf(stderr,"cuda_assert: %s %s %d\n",cudaGetErrorString(code),file,line);

        if (abort) {
            cudaDeviceReset();          
            exit(code);
        }
    }
}

#define cuda(...) { cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__, true ); }

static void sort(thrust::host_vector<int>& h_vec, cudaEvent_t start, cudaEvent_t end, float* const elapsed) {
    
    cuda(EventRecord (start));

    thrust::device_vector<int> d_vec = h_vec; // copy data to device
    thrust::sort(d_vec.begin(), d_vec.end()); // sort data on device 

    cuda(EventRecord(end));
    cuda(EventSynchronize(end));

    float sort_elapsed;
    cuda(EventElapsedTime(&sort_elapsed, start, end));
    *elapsed += sort_elapsed;
}

static void measure(const struct cudaDeviceProp* const props, const int DATASIZE) {
    
    thrust::host_vector<int> h_vec(DATASIZE);
    std::generate(h_vec.begin(), h_vec.end(), rand);

    cudaEvent_t start, end;
    cuda(EventCreate(&start));
    cuda(EventCreate(&end));

    float elapsed = 0.0f;
    for (int a=0; a<32; a++) sort(h_vec, start, end, &elapsed);

    cuda(EventDestroy(start));
    cuda(EventDestroy(end));

    float time = elapsed / 32.0 / 1000; // in secs
    printf("Throughput =%9.3lf MElements/s, Time = %.3lf ms\n\n", 
        1e-6 * DATASIZE / time, time * 1000);
}

int main(int argc, char** argv) {

    DisplayCudaDevice();
   
    struct cudaDeviceProp props;
    const int DATASIZE = atoi(argv[1]);
    printf("Sorting %d elements:\n", DATASIZE);
    
    measure( &props, DATASIZE );  // SORT
    cuda(DeviceReset()); // RESET

    return 0;
}

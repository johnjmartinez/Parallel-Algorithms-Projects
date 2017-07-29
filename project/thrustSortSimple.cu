// COMPILE: nvcc thrustSortSimple.cu -I cuda_common/inc/ -o thrustSort

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>

static void cuda_assert(const cudaError_t code, const char* const file, const int line, const bool abort) {
    if (code != cudaSuccess) {
        fprintf(stderr,"cuda_assert: %s %s %d\n",cudaGetErrorString(code),file,line);

        if (abort) {
            cudaDeviceReset();          
            exit(code);
        }
    }
}

#define cuda(...) { cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__, true); }

static void sort(thrust::host_vector<uint64_t>& h_vec, cudaEvent_t start, cudaEvent_t end, float* const elapsed) {
    
    thrust::device_vector<uint64_t> d_vec = h_vec; // copy data to device
    cuda(EventRecord(start,0));

    thrust::sort(d_vec.begin(), d_vec.end()); // sort data on device 

    cuda(EventRecord(end,0));
    cuda(EventSynchronize(end));

    float sort_elapsed;
    cuda(EventElapsedTime(&sort_elapsed,start,end));

    *elapsed += sort_elapsed;
}

static void measure(const struct cudaDeviceProp* const props, const uint32_t count) {
    
    thrust::host_vector<uint64_t> h_vec(count);
    std::generate(h_vec.begin(), h_vec.end(), rand);

    cudaEvent_t start, end;
    cuda(EventCreate(&start));
    cuda(EventCreate(&end));

    float elapsed = 0.0f;
    for (int a=0; a<20; a++) sort(h_vec,start,end,&elapsed);

    cuda(EventDestroy(start));
    cuda(EventDestroy(end));

    fprintf(stdout," elements:%u, time:%.2f\n", count, elapsed/20.0);
}

int main(int argc, char** argv) {

    const int32_t device = 0 ;
    struct cudaDeviceProp props;
    
    cuda(GetDeviceProperties(&props,device));
    printf("%s (%d)\n",props.name,props.multiProcessorCount);

    cuda(SetDevice(device));

    const uint32_t count = (32<<20); // 32M
    measure(&props,count);  // SORT
    cuda(DeviceReset()); // RESET

    return 0;
}

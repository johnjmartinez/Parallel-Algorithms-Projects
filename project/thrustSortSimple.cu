// COMPILE: nvcc thrustSortSimple.cu -I cuda_common/inc/ -o thrustSort

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

#include <stdbool.h>

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
  // transfer data to the device
  thrust::device_vector<uint64_t> d_vec = h_vec;

  cuda(EventRecord(start,0));
  
  // sort data on the device 
  thrust::sort(d_vec.begin(), d_vec.end());

  cuda(EventRecord(end,0));
  cuda(EventSynchronize(end));

  float sort_elapsed;
  cuda(EventElapsedTime(&sort_elapsed,start,end));

  *elapsed += sort_elapsed;
}

#define THRUST_SORT_WARMUP 5
#define THRUST_SORT_BENCH  100000

static void bench(const struct cudaDeviceProp* const props, const uint32_t count) {
  // generate 32M random numbers serially
  thrust::host_vector<uint64_t> h_vec(count);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  cudaEvent_t start, end;
  cuda(EventCreate(&start));
  cuda(EventCreate(&end));

  float warmup = 0.0f;

  for (int ii=0; ii<THRUST_SORT_WARMUP; ii++)
    sort(h_vec,start,end,&warmup);

  float elapsed = 0.0f;
  
  for (int ii=0; ii<THRUST_SORT_BENCH; ii++)
    sort(h_vec,start,end,&elapsed);

  cuda(EventDestroy(start));
  cuda(EventDestroy(end));

  fprintf(stdout,"%s, %u, %u, %.2f, %u, %.2f, %.2f\n",
          props->name, props->multiProcessorCount, count, elapsed, THRUST_SORT_BENCH, (double)elapsed/THRUST_SORT_BENCH, THRUST_SORT_BENCH*count/(elapsed*1000.0));
}

int main(int argc, char** argv) {
  const int32_t device = (argc == 1) ? 0 : atoi(argv[1]);

  struct cudaDeviceProp props;
  cuda(GetDeviceProperties(&props,device));

  printf("%s (%2d)\n",props.name,props.multiProcessorCount);

  cuda(SetDevice(device));

  const uint32_t count_lo   = argc <= 2 ? 16384     : strtoul(argv[2],NULL,0);
  const uint32_t count_hi   = argc <= 3 ? count_lo  : strtoul(argv[3],NULL,0);
  const uint32_t count_step = argc <= 4 ? 256       : strtoul(argv[4],NULL,0);

  // SORT
  for (uint32_t count=count_lo; count<=count_hi; count+=count_step)
    bench(&props,count);

  // RESET
  cuda(DeviceReset());

  return 0;
}

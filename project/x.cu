#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_common.h>     
#include <algorithm>
//#include <thrust/sort.h>
//#include <thrust/device_ptr.h>
#include <moderngpu/kernel_mergesort.hxx>

#define SHARED_SIZE_LIMIT 1024U
#define BLOCK_THREADS 32
#define ELEMS_PER_THREAD 32
#define LOOPS 32

using namespace mgpu;

void printArray(int *a, int len, const char *fileName) {
    FILE *fptr;
    fptr = fopen(fileName, "w");
    for (int i = 0; i < len; i++) fprintf(fptr, "%d\n", a[i]);
    fclose(fptr);
}

int main(int argc, char **argv) {

    //DisplayCudaDevice();
    standard_context_t context(false);
    
    float et = 0;
    float tmp_time = 0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    
    const int DATASIZE = atoi(argv[1]);
    unsigned int arrAlloc = DATASIZE * sizeof(int);
    printf("Sorting %d elements\n", DATASIZE); 

    srand(time(NULL));
	int *hdata, *ddata;
        
    hdata = (int*)malloc(arrAlloc);   // allocating memory for Hosts[]s
    for( int i = 0; i < DATASIZE; i++ ) hdata[i]  = rand() ; 
    //printArray(hdata, DATASIZE, "input");
    
    checkCudaErrors(cudaMalloc((void**)&ddata, arrAlloc)); // allocating memory for Dev[]s
    checkCudaErrors(cudaDeviceSynchronize());
	for (int i = 0; i < LOOPS ; i++) {
        checkCudaErrors(cudaEventRecord(start)); //start tmp_time
		
        checkCudaErrors(cudaMemcpy(ddata, hdata, arrAlloc, cudaMemcpyHostToDevice));
        //thrust::device_ptr<int> d_a(ddata);
        //thrust::sort(d_a, d_a+DATASIZE);
        mergesort(ddata, DATASIZE, less_t<int>(), context);

        checkCudaErrors(cudaEventRecord(stop)); // end tmp_time
        checkCudaErrors(cudaEventSynchronize(stop));

        tmp_time = 0;
        checkCudaErrors(cudaEventElapsedTime(&tmp_time, start, stop));
        et += tmp_time;
	}
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaMemcpy(hdata, ddata, arrAlloc, cudaMemcpyDeviceToHost));
    printf("Sorting %s\n", (std::is_sorted(hdata, hdata+DATASIZE) ? "succeed." : "FAILED.") );
    //printArray(hdata, DATASIZE, "output");
    
    tmp_time = et/1000/LOOPS;
    printf("Throughput =%9.3lf MElements/s, Time = %.3lf ms\n", 
        1e-6 * DATASIZE / tmp_time, tmp_time * 1000);

    if(ddata) cudaFree(ddata);
	return 0;
}

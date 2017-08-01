#include <cub/cub.cuh>
#include <cstdio>
#include <cuda.h>
#include <algorithm>
#include <cuda_common.h>       
#include "test/test_util.h"

using namespace cub;

template <typename T>
void cubDevSort (int COUNT, T *d_in, T *d_out) {
    
    int beginBit = 0, endBit = 64;//*sizeof(T); // Bit subrange [beginBit, endBit) for differentiating elem bits
    
    /*
    DoubleBuffer<T> dElems;
    cudaMalloc((void**)&dElems.d_buffers[0], COUNT*sizeof(T));
    cudaMalloc((void**)&dElems.d_buffers[1], COUNT*sizeof(T));
    */
    //auto d_array = d_in;
    DoubleBuffer<T> dElems(d_in, d_in+COUNT);
    
    void *dTmp = NULL;
    size_t dTmpSize = 0;
     // no work done and required allocation size is returned in dTmpSize.
    checkCudaErrors(DeviceRadixSort::SortKeys(dTmp, dTmpSize, dElems, COUNT, beginBit, endBit));
    checkCudaErrors(cudaMalloc (&dTmp, dTmpSize)); // Allocate temp storage
    
    // Run sorting operation
    cudaMemcpy(dElems.d_buffers[dElems.selector], d_in, sizeof(T) * COUNT, cudaMemcpyDeviceToDevice);
    checkCudaErrors(DeviceRadixSort::SortKeys(dTmp, dTmpSize, dElems, COUNT, beginBit, endBit));
     
    cudaMemcpy(d_out, dElems.Current(), sizeof(T) * COUNT, cudaMemcpyDeviceToDevice);
    checkCudaErrors(cudaFree(dTmp)); // Release temp storage
}

void printArray(int *a, int len, const char *fileName) {
    FILE *fptr;
    fptr = fopen(fileName, "w");
    for (int i = 0; i < len; i++) fprintf(fptr, "%d\n", a[i]);
    fclose(fptr);
}

int main(int argc, char** argv) {

    cudaDeviceReset();
    float et = 0;
    float tmp_time = 0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    const int DATASIZE = atoi(argv[1]);
    const int numIterations = 32;
    
    int *h_ary, *h_ref; // Host []
    int *d_1, *d_2; // Dev []
    
    unsigned int memAlloc = DATASIZE * sizeof(int);
    h_ary = (int*)malloc(memAlloc); // allocatings_output memory for Hosts[]
    h_ref = (int*)malloc(memAlloc); // allocatings_output memory for Hosts[]
    cudaMalloc((void**)&d_1, memAlloc+1024); // allocating memory for Dev[] + paddinng
    cudaMalloc((void**)&d_2, memAlloc+1024); // allocating memory for Dev[] + paddinng

    srand(time(NULL));
    for( int i = 0; i < DATASIZE; i++ ) h_ary[i]  = rand() ; // generating Host[] values
    //std::copy(h_ary, h_ary+DATASIZE, h_ref); std::sort(h_ref, h_ref+DATASIZE);
    
    printf("Input array size : %d\n", DATASIZE);
    //printArray(h_ary, DATASIZE, "input"); printArray(h_ref, DATASIZE, "s_input");

    checkCudaErrors(cudaDeviceSynchronize());
    for (unsigned int i = 0; i < numIterations; i++) {

        checkCudaErrors(cudaEventRecord(start)); //start tmp_time    
        checkCudaErrors(cudaMemcpy(d_1, h_ary, memAlloc, cudaMemcpyHostToDevice)); // copy from Host to Dev
        cubDevSort<int> (DATASIZE, d_1, d_2);
        checkCudaErrors(cudaEventRecord(stop)); // end tmp_time
        checkCudaErrors(cudaEventSynchronize(stop));

        tmp_time = 0;
        checkCudaErrors(cudaEventElapsedTime(&tmp_time, start, stop));
        et += tmp_time;
    } 
    checkCudaErrors(cudaDeviceSynchronize());
    
    tmp_time = et/1000/numIterations;    
    printf("Throughput =%9.3lf MElements/s, Time = %.3lf ms\n", 
        1e-6 * DATASIZE / tmp_time, tmp_time  * 1000);
    printf("int size: %lu\n", sizeof(int)*8);
    
    cudaMemcpy(h_ref, d_2, memAlloc, cudaMemcpyDeviceToHost); // copy from Dev to Host
    bool flag = std::is_sorted(h_ref, h_ref+DATASIZE);
    printf("Sorting %s\n", (flag)?"succeed.":"FAILED.");
    if (!flag) {
        printArray(h_ref, DATASIZE, "output");
        printArray(h_ary, DATASIZE, "input");
    }

    // Cleanup
    if(h_ary) delete[] h_ary;
    if(d_1) cudaFree(d_1);
    if(d_2) cudaFree(d_2);
    cudaDeviceReset();
    return 0;

}

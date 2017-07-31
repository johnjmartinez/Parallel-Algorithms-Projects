// https://stackoverflow.com/questions/21807872/making-cub-blockradixsort-on-chip-entirely

#include <cub/cub.cuh>
#include <cstdio>
#include <cuda.h>
#include <algorithm>
#include <cuda_common.h>       
#include "test/test_util.h"

#define BLOCK_THREADS 32
#define ELEMS_PER_THREAD 32
using namespace cub;

template <typename T>
__global__ void cubBlkSort(T *d_in, T *d_out) {

    enum { TILE_SIZE = BLOCK_THREADS * ELEMS_PER_THREAD };
    
    typedef BlockLoad<T, BLOCK_THREADS, ELEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> localBlkLoad;
    typedef BlockStore<T, BLOCK_THREADS, ELEMS_PER_THREAD, BLOCK_STORE_WARP_TRANSPOSE> localBlkStore;    
    typedef BlockRadixSort<T, BLOCK_THREADS, ELEMS_PER_THREAD> localBlkSort;
    
    __shared__ union {
        typename localBlkLoad::TempStorage load;
        typename localBlkStore::TempStorage store;
        typename localBlkSort::TempStorage sort;
    } tmpStorage;    

    T threadData[ELEMS_PER_THREAD];
    const int blkOffset = blockIdx.x * TILE_SIZE;
    
    localBlkLoad(tmpStorage.load).Load(d_in + blkOffset, threadData);
     __syncthreads(); 

    localBlkSort(tmpStorage.sort).Sort(threadData);
    __syncthreads(); 

    localBlkStore(tmpStorage.store).Store(d_out + blkOffset, threadData);
}

/* not working ....
template <typename T>
void cubDevSort (int SIZE, T *d_in, T *d_out) {
    
    int beginBit = 0, endBit = 8; // Bit subrange [beginBit, endBit) of differentiating elem bits

    // Determine temp Dev storage requirements
    void *dTmp = NULL;
    size_t dTmpSize = SIZE;
     // no work done and required allocation size is returned in dTmpSize.
    checkCudaErrors(DeviceRadixSort::SortKeys(dTmp, dTmpSize, d_in, d_out, SIZE)); // , beginBit, endBit));
    checkCudaErrors(cudaMalloc (&dTmp, dTmpSize)); // Allocate temp storage
    checkCudaErrors(cudaDeviceSynchronize());

    // Run sorting operation
    checkCudaErrors(DeviceRadixSort::SortKeys(dTmp, dTmpSize, d_in, d_out, SIZE)); // , beginBit, endBit) );
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(dTmp)); // Release temp storage
}*/

template <typename T>
void cubDevSort (int COUNT, T *d_in, T *d_out) {
    
    int beginBit = 0, endBit = 8*sizeof(T); // Bit subrange [beginBit, endBit) of differentiating elem bits
    
    DoubleBuffer<int> dElems;
    cudaMalloc((void**)&dElems.d_buffers[0], COUNT*sizeof(T));
    cudaMalloc((void**)&dElems.d_buffers[1], COUNT*sizeof(T));
    
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
    cudaMalloc((void**)&d_1, memAlloc*2); // allocating memory for Dev[] + paddinng
    cudaMalloc((void**)&d_2, memAlloc*2); // allocating memory for Dev[] + paddinng

    srand(time(NULL));
    for( int i = 0; i < DATASIZE; i++ ) h_ary[i]  = rand() ; // generating Host[] values
    //std::copy(h_ary, h_ary+DATASIZE, h_ref); std::sort(h_ref, h_ref+DATASIZE);
    
    printf("Input array size : %d\n", DATASIZE);
    //printArray(h_ary, DATASIZE, "input"); printArray(h_ref, DATASIZE, "s_input");
    int numBlks = DATASIZE / BLOCK_THREADS / ELEMS_PER_THREAD;
    
    checkCudaErrors(cudaDeviceSynchronize());
    for (unsigned int i = 0; i < numIterations; i++) {

        checkCudaErrors(cudaEventRecord(start)); //start tmp_time    
        checkCudaErrors(cudaMemcpy(d_1, h_ary, memAlloc, cudaMemcpyHostToDevice)); // copy from Host to Dev

        cubBlkSort<int> <<< numBlks, BLOCK_THREADS >>> (d_1, d_2);   
        cubDevSort<int> (DATASIZE, d_2, d_2);

        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaEventRecord(stop)); // end tmp_time
        checkCudaErrors(cudaEventSynchronize(stop));

        tmp_time = 0;
        checkCudaErrors(cudaEventElapsedTime(&tmp_time, start, stop));
        et += tmp_time;
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy(h_ref, d_2, memAlloc, cudaMemcpyDeviceToHost); // copy from Dev to Host
    printf("Sorting %s\n", (std::is_sorted(h_ref, h_ref+DATASIZE) ? "succeed." : "FAILED.") );
    //printArray(h_ref, DATASIZE, "output");
   
    tmp_time = et/1000/numIterations;    
    printf("Throughput =%9.3lf MElements/s, Time = %.3lf ms\n\n", 
        1e-6 * DATASIZE / tmp_time, tmp_time  * 1000);
    
    // Cleanup
    if(h_ary) delete[] h_ary;
    if(d_1) cudaFree(d_1);
    if(d_2) cudaFree(d_2);
    cudaDeviceReset();
    return 0;

}

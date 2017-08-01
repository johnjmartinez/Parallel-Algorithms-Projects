// https://stackoverflow.com/questions/21807872/making-cub-blockradixsort-on-chip-entirely

#include <cub/cub.cuh>
#include <cstdio>
#include <cuda.h>
#include <algorithm>
#include <cuda_common.h>       
#include "test/test_util.h"

using namespace cub;

template <typename T, int BlkThreads, int ItemsPerThread>
__global__ void cubBlkSort(T *d_in, int *d_out) {

    enum { TILE_SIZE = BlkThreads * ItemsPerThread };
    
    typedef BlockLoad<T, BlkThreads, ItemsPerThread, BLOCK_LOAD_WARP_TRANSPOSE> localBlkLoad;
    typedef BlockStore<T, BlkThreads, ItemsPerThread, BLOCK_STORE_WARP_TRANSPOSE> localBlkStore;    
    typedef BlockRadixSort<T, BlkThreads, ItemsPerThread> localBlkSort;
    
    __shared__ union {
        typename localBlkLoad::TempStorage load;
        typename localBlkStore::TempStorage store;
        typename localBlkSort::TempStorage sort;
    } tmpStorage;    

    int threadData[ItemsPerThread];
    const int blkOffset = blockIdx.x * TILE_SIZE;
    
    localBlkLoad(tmpStorage.load).Load(d_in + blkOffset, threadData);
     __syncthreads(); 

    localBlkSort(tmpStorage.sort).Sort(threadData);
    __syncthreads(); 

    localBlkStore(tmpStorage.store).Store(d_out + blkOffset, threadData);
}

template <typename T>
void cubDevSort (int COUNT, T *d_in, T *d_out) {
    
    int beginBit = 0, endBit = 32; // Bit subrange [beginBit, endBit) for differentiating elem bits
    DoubleBuffer<T> dElems(d_in, d_in+COUNT);
    
    void *dTmp = NULL;
    size_t dTmpSize = 0;
     // no work done and required allocation size is returned in dTmpSize.
    checkCudaErrors(DeviceRadixSort::SortKeys(dTmp, dTmpSize, dElems, COUNT, beginBit, endBit));
    checkCudaErrors(cudaMalloc (&dTmp, dTmpSize)); // Allocate temp storage
    
    // Run sorting operation
    checkCudaErrors(DeviceRadixSort::SortKeys(dTmp, dTmpSize, dElems, COUNT, beginBit, endBit));
     
    cudaMemcpy(d_out, dElems.Current(), sizeof(T)*COUNT, cudaMemcpyDeviceToDevice);
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
    int x = 5;
       
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    const int numIterations = 32;
    const int DATASIZE = atoi(argv[1]);
    if (argc == 3) x = atoi(argv[2]);
    
    int *h_ary, *h_ref; // Host []
    int *d_1, *d_2; // Dev []
    
    unsigned int memAlloc = DATASIZE * sizeof(int);
    h_ary = (int*)malloc(memAlloc); // allocatings_output memory for Hosts[]
    h_ref = (int*)malloc(memAlloc); // allocatings_output memory for Hosts[]
    cudaMalloc((void**)&d_1, memAlloc*8); // allocating memory for Dev[] + paddinng
    cudaMalloc((void**)&d_2, memAlloc*8); // allocating memory for Dev[] + paddinng

    printf("Input array size : %d\n", DATASIZE);
    srand(time(NULL));
    for( int i = 0; i < DATASIZE; i++ ) h_ary[i]  = rand() ; // generating Host[] values
    //std::copy(h_ary, h_ary+DATASIZE, h_ref); std::sort(h_ref, h_ref+DATASIZE);
    //printArray(h_ary, DATASIZE, "input"); printArray(h_ref, DATASIZE, "s_input");
    
    int BLOCK_THREADS = 1024>>x;
    int ELEMS_PER_THREAD = 1<<x;
    int numBlks = DATASIZE / BLOCK_THREADS / ELEMS_PER_THREAD;

    checkCudaErrors(cudaDeviceSynchronize());
    for (unsigned int i = 0; i < numIterations; i++) {

        checkCudaErrors(cudaEventRecord(start)); //start tmp_time    
        checkCudaErrors(cudaMemcpy(d_1, h_ary, memAlloc, cudaMemcpyHostToDevice)); // copy from Host to Dev
        switch(x) {
            case 0 : cubBlkSort<int, 1024,1> <<< numBlks, BLOCK_THREADS >>> (d_1, d_2); break;     
            case 1 : cubBlkSort<int, 512, 2> <<< numBlks, BLOCK_THREADS >>> (d_1, d_2); break;
            case 2 : cubBlkSort<int, 256, 4> <<< numBlks, BLOCK_THREADS >>> (d_1, d_2); break;     
            case 3 : cubBlkSort<int, 128, 8> <<< numBlks, BLOCK_THREADS >>> (d_1, d_2); break;     
            case 4 : cubBlkSort<int, 64, 16> <<< numBlks, BLOCK_THREADS >>> (d_1, d_2); break;     
            case 5 : cubBlkSort<int, 32, 32> <<< numBlks, BLOCK_THREADS >>> (d_1, d_2); break;     
        }
        cubDevSort<int> (DATASIZE, d_2, d_2);

        checkCudaErrors(cudaEventRecord(stop)); // end tmp_time
        checkCudaErrors(cudaEventSynchronize(stop));

        tmp_time = 0;
        checkCudaErrors(cudaEventElapsedTime(&tmp_time, start, stop));
        et += tmp_time;
    }    
    checkCudaErrors(cudaDeviceSynchronize());
   
    tmp_time = et/1000/numIterations;    
    printf("%4d Ts | %2d elems/T: Throughput =%9.3lf MElements/s, Time = %.3lf ms\n",
               BLOCK_THREADS, ELEMS_PER_THREAD, 1e-6 * DATASIZE / tmp_time, tmp_time * 1000);

    cudaMemcpy(h_ref, d_2, memAlloc, cudaMemcpyDeviceToHost); // copy from Dev to Host
    bool flag = std::is_sorted(h_ref, h_ref+DATASIZE);
    printf("Sorting %s\n", (flag)?"succeed.":"FAILED.");
    if (!flag){
        char buffer [20];
        sprintf (buffer, "%04d_%02d_output", BLOCK_THREADS, ELEMS_PER_THREAD);
        printArray(h_ref, DATASIZE, buffer);
    }
    
    // Cleanup
    if(h_ary) delete[] h_ary;
    if(d_1) cudaFree(d_1);
    if(d_2) cudaFree(d_2);
    cudaDeviceReset();
    return 0;

}

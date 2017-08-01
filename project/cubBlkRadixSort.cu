// https://stackoverflow.com/questions/21807872/making-cub-blockradixsort-on-chip-entirely

#include <cub/cub.cuh>
#include <cstdio>
#include <cuda.h>
#include <algorithm>
#include <cuda_common.h>       
#include "test/test_util.h"

using namespace cub;

template <typename T, int BLOCK_THREADS, int ELEMS_PER_THREAD>
__global__ void cubBlkSort(T *d_in, int *d_out) {

    enum { TILE_SIZE = BLOCK_THREADS * ELEMS_PER_THREAD };
    
    typedef BlockLoad<T, BLOCK_THREADS, ELEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> localBlkLoad;
    typedef BlockStore<T, BLOCK_THREADS, ELEMS_PER_THREAD, BLOCK_STORE_WARP_TRANSPOSE> localBlkStore;    
    typedef BlockRadixSort<T, BLOCK_THREADS, ELEMS_PER_THREAD> localBlkSort;
    
    __shared__ union {
        typename localBlkLoad::TempStorage load;
        typename localBlkStore::TempStorage store;
        typename localBlkSort::TempStorage sort;
    } tmpStorage;    

    int threadData[ELEMS_PER_THREAD];
    const int blkOffset = blockIdx.x * TILE_SIZE;
    
    localBlkLoad(tmpStorage.load).Load(d_in + blkOffset, threadData);
     __syncthreads(); 

    localBlkSort(tmpStorage.sort).Sort(threadData);
    __syncthreads(); 

    localBlkStore(tmpStorage.store).Store(d_out + blkOffset, threadData);
}

template <typename T>
void cubDevSort (int COUNT, int *d_in, int *d_out) {
    
    int beginBit = 0, endBit = 8*sizeof(T); // Bit subrange [beginBit, endBit) of differentiating int bits
    
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

void sort (const int x, const int SIZE, int *d_1, int *d_2, int *h_ary, int *h_ref,
            cudaEvent_t &start, cudaEvent_t &stop) {

    const int numIterations = 32;
    const int BLOCK_THREADS = 1024>>x;
    const int ELEMS_PER_THREAD = 1<<x;
    const int numBlks = SIZE / BLOCK_THREADS / ELEMS_PER_THREAD;
    
    float et = 0;
    float tmp_time = 0;
    unsigned int memAlloc = SIZE * sizeof(int);
    
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
        //cubDevSort<int> (SIZE, d_2, d_2);
        checkCudaErrors(cudaEventRecord(stop)); // end tmp_time
        checkCudaErrors(cudaEventSynchronize(stop));

        tmp_time = 0;
        checkCudaErrors(cudaEventElapsedTime(&tmp_time, start, stop));
        et += tmp_time;
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    tmp_time = et/1000/numIterations;        
    printf("%4d Ts | %2d elems/T: Throughput =%9.3lf MElements/s, Time = %.3lf ms\n",
               BLOCK_THREADS, ELEMS_PER_THREAD, 1e-6 * SIZE / tmp_time, tmp_time * 1000);

    printArray(h_ref, SIZE, "s_input"); printArray(h_ary, SIZE, "input");
    cudaMemcpy(h_ref, d_1, memAlloc, cudaMemcpyDeviceToHost); // copy from Dev to Host
    printf("Sorting %s\n", (std::is_sorted(h_ref, h_ref+SIZE) ? "succeed." : "FAILED.") );
    char buffer [20];
    sprintf (buffer, "%04d_%02d_output", BLOCK_THREADS, ELEMS_PER_THREAD);
    printArray(h_ref, SIZE, buffer);

}

int main(int argc, char** argv) {

    cudaDeviceReset();
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    DisplayCudaDevice();

    const int DATASIZE = atoi(argv[1]);
    printf("Input array size : %d\n", DATASIZE);
    
    int *h_ary, *h_ref; // Host []
    int *d_1, *d_2; // Dev []
    
    unsigned int memAlloc = DATASIZE * sizeof(int);
    
    h_ary = (int*)malloc(memAlloc); // allocatings_output memory for Hosts[]
    h_ref = (int*)malloc(memAlloc); // allocatings_output memory for Hosts[]
    cudaMalloc((void**)&d_1, memAlloc*2); // allocating memory for Dev[] + paddinng
    cudaMalloc((void**)&d_2, memAlloc*2); // allocating memory for Dev[] + paddinng

    srand(time(NULL));
    for( int i = 0; i < DATASIZE; i++ ) h_ary[i]  = rand() ; // generating Host[] values
    std::copy(h_ary, h_ary+DATASIZE, h_ref); std::sort(h_ref, h_ref+DATASIZE);    
    
    //for(int i=0; i<=5; i++) sort(i, DATASIZE, d_1, d_2, h_ary, h_ref, start, stop);  
    sort(4, DATASIZE, d_1, d_2, h_ary, h_ref, start, stop); 

    // Cleanup
    if(h_ary) delete[] h_ary;
    if(d_1) cudaFree(d_1);
    if(d_2) cudaFree(d_2);
    cudaDeviceReset();
    return 0;

}

#include <iostream>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <cuda_common.h>

#define BLOCK_SIZE 1024

#define CUDA_CHECK(val) cudaCheck( (val), #val, __FILE__, __LINE__)

template<typename T>
void cudaCheck(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

__global__ void checkBit(int* const d_inVals, int* const d_outPredct, const int bit, const size_t numElems) {
    // Predicate returns TRUE if significant bit is not present
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems) return;
    int Predct = ((d_inVals[id] & bit) == 0);
    d_outPredct[id] = Predct;
}

__global__ void flipBit(int* const d_list, const size_t numElems) { 
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems) return;
    d_list[id] = ((d_list[id] + 1) % 2);
}

__global__ void partnExclBlellochScan(int* const d_list, int* const d_blk_sums,  const size_t numElems) { 
    extern __shared__ int s_block_scan[];
    const int tid = threadIdx.x;
    const int id = blockDim.x * blockIdx.x + tid;

    // copy to shared memory, pad block if too small
    if (id >= numElems) s_block_scan[tid] = 0;
    else s_block_scan[tid] = d_list[id];
    
    __syncthreads();

    // reduce
    int i;
    for (i = 2; i <= blockDim.x; i <<= 1) {
      if ((tid + 1) % i == 0) {
        int neighbor_offset = i>>1;
        s_block_scan[tid] += s_block_scan[tid - neighbor_offset];
      }
      __syncthreads();
    }
    i >>= 1; // return i to last value before for-loop exit
    
    if (tid == (blockDim.x-1)) {
      d_blk_sums[blockIdx.x] = s_block_scan[tid];
      s_block_scan[tid] = 0; // set last (sum of whole block) to 0
    }
    
    __syncthreads();

    // downsweep
    for (i = i; i >= 2; i >>= 1) {
      if((tid + 1) % i == 0) {
        int neighbor_offset = i>>1;
        int old_neighbor = s_block_scan[tid - neighbor_offset];
        s_block_scan[tid - neighbor_offset] = s_block_scan[tid]; // copy
        s_block_scan[tid] += old_neighbor;
      }
      __syncthreads();
    }

    // copy result to global memory
    if (id < numElems) d_list[id] = s_block_scan[tid];
}

__global__ void scanAddBlkSums(int* const d_predctScan, int* const d_blkSumScan, const size_t numElems) { 
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems) return;
    d_predctScan[id] += d_blkSumScan[blockIdx.x];
}

__global__ void scatter(int* const d_in, int* const d_out, int* const d_predctTscan, int* const d_predctFscan,
                        int* const d_predctFalse, int* const d_numPredctTelems, const size_t numElems) { 

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems) return;

    int newLoc;
    if (d_predctFalse[id] == 1)  newLoc = d_predctFscan[id] + *d_numPredctTelems;
    else newLoc = d_predctTscan[id];

    d_out[newLoc] = d_in[id];
}

int* d_predct;
int* d_predctTscan;
int* d_predctFscan;
int* d_numPredctTelems; // GLOBAL POINTERS
int* d_numPredctFelems;
int* d_blk_sums;

void radixSort(int* const d_inVals, int* const d_inPos, int* const d_outVals, int* const d_outPos, 
               const size_t numElems) { 

  int blockSize = BLOCK_SIZE;
  size_t size = sizeof(int) * numElems;
  int gridSize = ceil(float(numElems) / float(blockSize));

  CUDA_CHECK(cudaMalloc((void**)&d_predct, size));
  CUDA_CHECK(cudaMalloc((void**)&d_predctTscan, size));
  CUDA_CHECK(cudaMalloc((void**)&d_predctFscan, size));
  CUDA_CHECK(cudaMalloc((void**)&d_numPredctTelems, sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&d_numPredctFelems, sizeof(int))); // throwaway
  CUDA_CHECK(cudaMalloc((void**)&d_blk_sums, gridSize*sizeof(int)));

  size_t blkAlloc = sizeof(int)*blockSize;
  size_t gridAlloc = sizeof(int)*gridSize;

  int bitPosition;
  int max_bits = 31;
  for (int bit = 0; bit < max_bits; bit++) {
    bitPosition = 1<<bit;

    // create PredctTrue
    if ((bit + 1) % 2 == 1) 
      checkBit<<<gridSize, blockSize>>>(d_inVals, d_predct, bitPosition, numElems);
    else 
      checkBit<<<gridSize, blockSize>>>(d_outVals, d_predct, bitPosition, numElems);
    
    // scan Predct=True
    CUDA_CHECK(cudaMemcpy(d_predctTscan, d_predct, size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(d_blk_sums, 0, gridAlloc));

    partnExclBlellochScan<<<gridSize, blockSize, blkAlloc>>>(d_predctTscan, d_blk_sums, numElems);
    partnExclBlellochScan<<<1, blockSize, gridAlloc>>>(d_blk_sums, d_numPredctTelems, gridSize);
    scanAddBlkSums<<<gridSize, blockSize>>>(d_predctTscan, d_blk_sums, numElems);

    // transform Predct=True -> Predct=False
    flipBit<<<gridSize, blockSize>>>(d_predct, numElems);

    // scan Predct=False
    CUDA_CHECK(cudaMemcpy(d_predctFscan, d_predct, size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(d_blk_sums, 0, gridAlloc));

    partnExclBlellochScan<<<gridSize, blockSize, blkAlloc>>>(d_predctFscan, d_blk_sums, numElems);
    partnExclBlellochScan<<<1, blockSize, gridAlloc>>>(d_blk_sums, d_numPredctFelems, gridSize);
    scanAddBlkSums<<<gridSize, blockSize>>>(d_predctFscan, d_blk_sums, numElems);

    // scatter values (flip input/output depending on iteration)
    if ((bit + 1) % 2 == 1) {
      scatter<<<gridSize, blockSize>>>(d_inVals, d_outVals, d_predctTscan, d_predctFscan, d_predct, d_numPredctTelems, numElems);
      scatter<<<gridSize, blockSize>>>(d_inPos, d_outPos, d_predctTscan, d_predctFscan, d_predct, d_numPredctTelems, numElems);
    } 
    else {
      scatter<<<gridSize, blockSize>>>(d_outVals, d_inVals, d_predctTscan, d_predctFscan, d_predct, d_numPredctTelems, numElems);
      scatter<<<gridSize, blockSize>>>(d_outPos, d_inPos, d_predctTscan, d_predctFscan, d_predct, d_numPredctTelems, numElems);
    }
  }

  CUDA_CHECK(cudaFree(d_predct));
  CUDA_CHECK(cudaFree(d_predctTscan));
  CUDA_CHECK(cudaFree(d_predctFscan));
  CUDA_CHECK(cudaFree(d_numPredctTelems));
  CUDA_CHECK(cudaFree(d_numPredctFelems));
  CUDA_CHECK(cudaFree(d_blk_sums));
}

void printArray(int *a, int len, const char *fileName) {
    FILE *fptr;
    fptr = fopen(fileName, "w");
    for (int i = 0; i < len; i++) fprintf(fptr, "%d\n", a[i]);
    fclose(fptr);
}

int main(int argc, char** argv)  {
    
    const int DATASIZE = atoi(argv[1]); 
    const int numIterations = 1;

    float et = 0;
    float tmp_time = 0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    
    int *h_data, *h_pos; // Host []s
    int *d_data, *d_pos, *d_data_out, *d_pos_out; // Dev []s
    unsigned int arrAlloc = DATASIZE * sizeof(int);
    
    h_data = (int*)malloc(arrAlloc);   // allocating memory for Hosts[]s
    h_pos = (int*)malloc(arrAlloc);

    cudaMalloc((void**)&d_data, arrAlloc); // allocating memory for Dev[]s 
    cudaMalloc((void**)&d_pos, arrAlloc); 
    cudaMalloc((void**)&d_data_out, arrAlloc); 
    cudaMalloc((void**)&d_pos_out, arrAlloc); 

    srand(time(NULL));
    for( int i = 0; i < DATASIZE; i++ ) {  // generating Host[] values
        h_data[i]  = rand() ; h_pos[i] = i;
    }
    printf("Sorting %d elements\n", DATASIZE); //DEBUG: printArray(h_data, DATASIZE, "input");
    
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos, arrAlloc, cudaMemcpyHostToDevice));  // copy from Host to Dev
    cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
    for (unsigned int i = 0; i < numIterations; i++) {

        cudaEventRecord(start);    
        CUDA_CHECK(cudaMemcpy(d_data, h_data, arrAlloc, cudaMemcpyHostToDevice));  // copy from Host to Dev

        radixSort(d_data, d_pos, d_data_out, d_pos_out, DATASIZE);

        cudaEventRecord(stop);    
        cudaEventSynchronize(stop); 
        cudaEventElapsedTime(&tmp_time, start, stop);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        et += tmp_time;
    }

    CUDA_CHECK(cudaMemcpy(h_data, d_data_out, arrAlloc, cudaMemcpyDeviceToHost));  // copy from Dev to Host
    printf("Sorting %s\n", (std::is_sorted(h_data, h_data+DATASIZE) ? "succeed." : "FAILED.") );

/* DEBUG:
    CUDA_CHECK(cudaMemcpy(h_pos, d_pos_out, arrAlloc, cudaMemcpyDeviceToHost));  // copy from Dev to Host
    printArray(h_data, DATASIZE, "data_out"); printArray(h_pos, DATASIZE, "post_out");
*/
    tmp_time = et/1000/numIterations;
    printf("Throughput =%9.3lf MElements/s, Time = %.9lf ms\n",  1e-6*DATASIZE/tmp_time, tmp_time*1000);

    cudaFree(d_data);
    cudaFree(d_pos);
    cudaDeviceReset();
    return 0;
}

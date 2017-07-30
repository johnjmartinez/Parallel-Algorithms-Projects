#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_common.h>     

#define SHARED_SIZE_LIMIT 1024U

//Map to single instructions on G8x / G9x / G100
#define UMUL(a, b) __umul24((a), (b))
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

__device__  void compare_and_swap( int &_a_, int &_b_) {
    int tmp;
    if ((_a_ > _b_)) {
        tmp = _a_;
        _a_ = _b_;
        _b_ = tmp;
    }
}

__global__ void oddEvenSortShared( int *d_input, int *d_output, int d_size ) {
    
    __shared__ int s_tmp[SHARED_SIZE_LIMIT];

    // offset to beginning 
    d_input  += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_output += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    
    s_tmp[threadIdx.x] = d_input[0];     // load data
    s_tmp[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_input[(SHARED_SIZE_LIMIT / 2)];

    for (int size = 2; size <= d_size; size <<= 1) {
    
        int stride = size / 2;
        int offset = threadIdx.x & (stride - 1);
        
        {
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            compare_and_swap( s_tmp[pos], s_tmp[pos + stride] );
            stride >>= 1;
        }

        for (; stride > 0; stride >>= 1) {
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            if (offset >= stride) compare_and_swap( s_tmp[pos - stride], s_tmp[pos] );
        }
    }
    __syncthreads();
    
    d_output[0] = s_tmp[threadIdx.x]; // copy to output
    d_output[(SHARED_SIZE_LIMIT / 2)] = s_tmp[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}


__global__ void oddEvenMergeGlobal( int *d_input, int *d_output, int size, int stride ) {
    
    int globalComptrIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = 2 * globalComptrIdx - (globalComptrIdx & (stride - 1)); //Odd-even merge

    if (stride < size / 2) {
        int offset = globalComptrIdx & ((size / 2) - 1);
        if (offset >= stride) {
            int elemA = d_input[pos - stride];
            int elemB = d_input[pos];

            compare_and_swap( elemA, elemB );

            d_output[pos - stride] = elemA;
            d_output[pos] = elemB;
        }
    }
    else {
        int elemA = d_input[pos];
        int elemB = d_input[pos + stride];

        compare_and_swap( elemA, elemB );

        d_output[pos] = elemA;
        d_output[pos + stride] = elemB;
    }
}

void printArray(int *a, int len, char *fileName) {
    FILE *fptr;
    fptr = fopen(fileName, "w");
    for (int i = 0; i < len; i++) fprintf(fptr, "%d\n", a[i]);
    fclose(fptr);
}


int main()  {

    DisplayCudaDevice();
    
    float et = 0;
    float tmp_time = 0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    
    unsigned int DATASIZE = 8<<20; 
    unsigned int numIterations = 20;
    
    int *h_input, *h_output; // Host []s
    int *d_input, *d_output; // Dev []s
    
    unsigned int arr_size = DATASIZE * sizeof(int);
    
    h_input = (int*)malloc(arr_size);   // allocating memory for Hosts[]s
    h_output = (int*)malloc(arr_size);
    checkCudaErrors( cudaMalloc((void**)&d_input, arr_size) ); // allocating memory for Dev[]s
    checkCudaErrors( cudaMalloc((void**)&d_output, arr_size) );

    srand(time(NULL));
    for( int i = 0; i < DATASIZE; i++ ) {
        h_input[i]  = rand() ; // generating Host[] values
        h_output[i] = 0;
    }
    printf("Input array size : %d:\n", DATASIZE);
    //printArray(h_input, DATASIZE, "input");
    
    int nblocks = DATASIZE / SHARED_SIZE_LIMIT;
    int nthreads = SHARED_SIZE_LIMIT / 2;
    
    checkCudaErrors( cudaDeviceSynchronize() );

    for (unsigned int i = 0; i < numIterations; i++) {

        checkCudaErrors( cudaEventRecord(start, nullptr) ); //start tmp_time
        checkCudaErrors( cudaMemcpy( d_input, h_input, arr_size, cudaMemcpyHostToDevice) ); // copy from Host to Dev

        oddEvenSortShared<<< nblocks, nthreads >>>( d_input, d_output, SHARED_SIZE_LIMIT ); // sort on shared

        for (int size = 2 * SHARED_SIZE_LIMIT; size <= DATASIZE; size <<= 1) {
            for (int stride = size / 2; stride > 0; stride >>= 1) {
                oddEvenMergeGlobal<<< DATASIZE / 512, 256 >>>( d_output, d_output, size, stride ); // merge on global
            }
        }
        
        checkCudaErrors( cudaEventRecord(stop, nullptr) ); // end tmp_time
        checkCudaErrors( cudaEventSynchronize(stop) );

        tmp_time = 0;
        checkCudaErrors( cudaEventElapsedTime(&tmp_time, start, stop) );
        et += tmp_time;
    }
    
    tmp_time = et/1000;
    printf("Throughput =%9.3lf MElements/s, Time = %.3lf ms\n", 1e-6 * DATASIZE / tmp_time, tmp_time*1000);

    checkCudaErrors( cudaDeviceSynchronize() );
    checkCudaErrors( cudaMemcpy( h_output, d_output, arr_size, cudaMemcpyDeviceToHost) ); // copy from Dev to Host
    //printArray(h_output, DATASIZE, "output");

    for(int i=0; i<DATASIZE-1; i++) { // CHECK --- like this a lot
        if(h_output[i] > h_output[i+1]) {
            printf("Sorting Failed\n");
            break;
        }
        else if(i == DATASIZE-2) // very ninja
            printf("Sorting Success\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaDeviceReset();
    return 0;
}

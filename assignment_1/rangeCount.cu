#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>

using namespace std;

/*
-- Counting nums in specific ranges
2. (40 pts) Read an array A as in the first question.
    (a, 10 pts) Create an array B of size 10 that keeps a count of the entries in each of the ranges: [0,99], [100,199], [200,299], ... Maintain an array B in global memory of GPU.
    
    (b, 10 pts) Repeat part (a) but first use the shared memory in a block for updating the local copy of B in each block. Once every block is done, add all local copies to get the global copy of B.
    
    (c, 20 pts) Create an array of size 10 that uses B to compute C which keeps count of the entries in each of the ranges: [0,99], [0,199], [0,299], ... , [0,999]. Note that the ranges are different from part (a). DO NOT use array A.
*/    

__global__
void countRangesGlobal(int size, int *A, int *B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    int x = A[i] / 100; // Find position x in B using positive truncation
    atomicAdd(&B[x], 1);
}

__global__
void countRangesShared(int size, int *A, int *B) {
    __shared__ int tmp[10];
    
    int tGlobalId = blockIdx.x * blockDim.x + threadIdx.x;
    int tId = threadIdx.x;
    if (tGlobalId >= size) return;
    int x = A[tGlobalId] / 100; // Find position x in B using positive truncation
    
    if (tId < 10) 
        tmp[tId] = 0; // 0 init share tmp mem
    __syncthreads();
    
    atomicAdd(&tmp[x], 1);
    __syncthreads();

    if (tId < 10)
        atomicAdd(&B[tId],tmp[tId]);
}

__global__
void scan (int size, int *B, int *C) {
     __shared__ int tmp[10];
    int curr = threadIdx.x;
    tmp[curr] = B[curr];
    
    for (int i = 1; i < size; i <<= 1) {
        __syncthreads();
        if (curr >= i)  tmp[curr] += tmp[curr-i];
    }
    
    C[curr] = tmp[curr];
     __syncthreads();
}

// output contents of array to screen
void printArray(int arr[], int size) {
    for ( int i = 0; i < size; i++ ) {
        cout << arr[i] << '\t';
    }
    cout << endl ;
}

int main(int argc, char *argv[]) {

    // <program> <file> <num of elems in file> -- argc should be 3 for correct execution 
    if ( argc != 3 ) { 
        printf( "usage: %s <file> <num of elem in file>\n", argv[0] );
        return -1;
    }
    
    int count = atoi(argv[2]);    
    ifstream InFile(argv[1]);
    stringstream buffer;
    buffer << InFile.rdbuf();
    //cout << tmp << "\n";  // DEBUG
    
    int a[count], b[10] = {}, c[10] = {};

    // READ FILE INTO a[]
    int x = 0;
    size_t last = 0, next;    
    string tmp = buffer.str();
    string delim = ",";
    while ( (next = tmp.find(delim, last)) != string::npos) { 
        a[x] = stoi( tmp.substr(last, next-last) );
        last = next + 1; 
        x++;
    } 
    a[x] = stoi( tmp.substr(last) ); // compile using nvcc -std=c++11 

    // Allocate space for device copies of a, b, c 
    int *A, *B, *C;
    int int_size = sizeof(int);
    int ten_size = 10*int_size;
    cudaMallocManaged(&A, count*int_size);
    cudaMallocManaged(&B, ten_size);
    
    // Copy inputs to device
    cudaMemcpy(A, &a, count*int_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B, &b, ten_size, cudaMemcpyHostToDevice); 
       
    int blockSize = 1024;
    int numBlocks = (count + blockSize - 1) / blockSize;
    
    // Q2a
    countRangesGlobal<<<numBlocks, blockSize>>>(count, A, B);
    cudaDeviceSynchronize();  // Wait for GPU to finish
    cudaMemcpy(&b, B, ten_size, cudaMemcpyDeviceToHost); 
    printArray(b, 10); // DEBUG
    cudaFree(B);    

    // Q2b
    for (int i=0; i<10; i++) { b[i] = 0; }   
    cudaMallocManaged(&B, ten_size);
    cudaMemcpy(B, &b, ten_size, cudaMemcpyHostToDevice); 
    countRangesShared<<<numBlocks, blockSize>>>(count, A, B);
    cudaDeviceSynchronize();  // Wait for GPU to finish
    cudaMemcpy(&b, B, ten_size, cudaMemcpyDeviceToHost); 
    printArray(b, 10); // DEBUG

    // Q2c
    cudaFree(A);
    cudaMallocManaged(&C, ten_size);
    scan<<<1, 10, ten_size>>>(10, B, C);
    cudaDeviceSynchronize();  // Wait for GPU to finish
    
    cudaMemcpy(&b, B, ten_size, cudaMemcpyDeviceToHost); 
    cudaMemcpy(&c, C, ten_size, cudaMemcpyDeviceToHost); 
    cudaFree(B);    
    cudaFree(C);    
    
    printArray(c, 10); // DEBUG
    
    return 0;
  
} //END MAIN

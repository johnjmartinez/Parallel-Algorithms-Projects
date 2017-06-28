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
void countRanges (int count, int *A, int *B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int x = A[i] / 100; // Find position in B using positive truncation
    //printf("x:%d\tAi:%d\ti:%d\tblk:%d\tthr:%d\n", x, A[i], i, blockIdx.x, threadIdx.x); // DEBUG 
    atomicAdd(&B[x], 1);
}

void scan (int size, int *B, int *C) {
    for(int i=0; i < size ; i++) {
        if (i==0) C[i] = B[i];
        else C[i] = B[i] + C[i-1];
    }
}

// output contents of array to screen
void printArray(int arr[], int size) {
    for ( int i = 0; i < size; i++ ) {
        cout << arr[i] << '\t';
    }
    cout << endl << endl;
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
    
    int a[count], b[10] = {0}, c[10] = {0};

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
    int size = sizeof(int);
    int ten_size = 10*size;
    cudaMallocManaged(&A, count*size);
    cudaMallocManaged(&B, ten_size);
    cudaMallocManaged(&C, ten_size);
    
    // Copy inputs to device
    cudaMemcpy(A, &a, count*size, cudaMemcpyHostToDevice);
    cudaMemcpy(B, &b, ten_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(C, &c, ten_size, cudaMemcpyHostToDevice); 
    
    //printArray(a, count); // DEBUG
    //printArray(A, count); // DEBUG
    
    int blockSize = 1024;
    int numBlocks = (count + blockSize - 1) / blockSize;
    countRanges<<<numBlocks, blockSize>>>(count, A, B);
    cudaDeviceSynchronize();  // Wait for GPU to finish
    
    // Copy result back to host
    cudaMemcpy(&b, B, ten_size, cudaMemcpyDeviceToHost); 
    printArray(b, 10); // DEBUG

    //scan(10, B, C);
    //cudaDeviceSynchronize();  // Wait for GPU to finish
    //printArray(c, 10); // DEBUG
    
    // FREE MEM
    cudaFree(A);
    cudaFree(B);    
    cudaFree(C);    

    return 0;
  
} //END MAIN


// http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
// http://people.ds.cam.ac.uk/pmb39/GPULectures/Lecture_2.pdf
// http://cuda-programming.blogspot.com/2013/03/computing-histogram-on-cuda-cuda-code_8.html
// https://stackoverflow.com/questions/15782325/cuda-programming-histogram
// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/histogram64/doc/histogram.pdf
// http://www.drdobbs.com/parallel/a-robust-histogram-for-massive-paralleli/240161600
// http://15418.courses.cs.cmu.edu/spring2017/
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
// https://link.springer.com/chapter/10.1007%2F978-3-642-37410-4_23

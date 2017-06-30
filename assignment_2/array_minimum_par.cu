#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define nthreads 450 
#define nblocks 1

/*
1. Write a parallel program in CUDA that reads a text file 'inp.txt'that contains a list of integers in the range [0-999] separated by commas. Your program should read this file in an array A of integers of size n.
    (a, 10 pts) Compute minA, the minimum value in the array. Your algorithm should take O(log n) time.
*/

// Parallel GPU function
__global__ void gpu_min(int *a, int *d ) {

    unsigned int tid = threadIdx.x;
    // unsigned int myId = blockIdx.x+blockDim.x+threadIdx.x;

    //dividing the number of threads by 2, if s=10, 0th tid will be compared with 10th tid and min will reside in tid 0. thus nlogn time complexity.
    for(unsigned int s=nthreads/2 ; s >= 1 ; s=s/2) {
	    if(tid < s) {
		    if(a[tid] > a[tid + s]) 
			    a[tid] = a[tid + s];
	    }
	    __syncthreads();
    }

    if(tid == 0 ) {
	    d[blockIdx.x] =a[0];
    }
}


//sequential CPU function
int cpu_min(int D[], int n) {
    int min = D[0], i;
    for(i = 1; i < n; i++) {
        if(min > D[i])
            min = D[i];
    }
    return min;
}


int main(int argc, char *argv[]) {

    // here I am giving a command line arg prg name inp.txt n where n is the number of records in the file.
    if ( argc != 3 ) /* argc should be 3 for correct execution */ {
        /* We print argv[0] assuming it is the program name */
        printf( "usage: %s file1", argv[0] );
        return 0;
    }
    else {
        int num_elements = atoi(argv[2]);
	    const int N=nthreads*nblocks;
	    const int arr_size = N*sizeof(int);


        int *h_input = (int*)malloc(arr_size);
        int i;

        FILE * pFile1;
        pFile1 = fopen (argv[1],"r");

        if (pFile1 == 0) {
            printf("could not open file\n");
            printf(" Error %d", errno );
            // return 0;
        }

        if(pFile1!=NULL) {
            for (i = 0; i < num_elements; i++) {
                fscanf(pFile1, "%d,", &(h_input[i]));
            }
        }
        
        printf("Number of elements in the array: %d \n ", num_elements);          
        printf("Input Array is: ");
        for(i = 0; i < num_elements; i++) {
            printf( "%d ", h_input[i]);
        }

	    fclose(pFile1);

        //host arrays
        int *h_output;

        //allocate memory for the host arrays
	    h_output = (int*)malloc(arr_size);
	    
        //device arrays
        int *d_input,*d_output;
        
        //allocate memory for device arrays
        cudaMalloc((void **) &d_input, arr_size);
        cudaMalloc((void **) &d_output, arr_size);
        
        cudaMemcpy(d_input, h_input, arr_size, cudaMemcpyHostToDevice);
        
        //GPU min call
        gpu_min <<< nblocks, nthreads >>>(d_input, d_output);
        cudaMemcpy(h_output, d_output, arr_size, cudaMemcpyDeviceToHost);
        
        //CPU min call
        int min = cpu_min(h_input, num_elements);
        
        printf("\n");
        printf(" Sequential CPU: Min number is %d \n ", min);
        printf(" Parallel GPU: Min number is %d \n", h_output[0]);
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        return 0;
    }//END ELSE
}//END MAIN

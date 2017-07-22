#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>


#define nthreads 639587
#define nblocks 625
/*
1. Write a parallel program in CUDA that reads a text file 'inp.txt'that contains a list of integers in the range [0-999] separated by commas. Your program should read this file in an array A of integers of size n.
    (b, 10 pts) Compute an array B such that B[i] is the last digit of A[i] for all i. Your algorithm should take O(1) time assuming n processors.
*/
    
// Parallel GPU function
__global__ void GPU_output(int *d_input, int *d_output) {
    
    //unsigned int tid = threadIdx.x;
    unsigned int divisor =1;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    d_output[i] = (d_input[i] / divisor) % 10;    
}

//sequential CPU function
void CPU_output(int D[], int n) {

    int i, divisor =1;
    int *output_arr = (int *)calloc(n, sizeof(int));
    
    for(i = 0; i < n; i++) {
        output_arr[i] = (D[i] / divisor) % 10;
    }

    printf( "Output Array : CPU : ");
    for(i = 0; i < n; i++) {
        printf(  "%d ", output_arr[i]);
    }

    printf( "\n");
}

int main(int argc, char *argv[]) {

    // here I am giving a command line arg prg name inp.txt n where n is the number of records in the file.
    if ( argc != 3 ) /* argc should be 3 for correct execution */ {
        /* We print argv[0] assuming it is the program name */
        printf( "usage: %s file1", argv[0] );
        return -1;
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
            //printf("%d\n", num_elements );
            for (i = 0; i < num_elements; i++) {
                fscanf(pFile1, "%d,", &(h_input[i]));
            }
        }

        printf("Number of elements: %d \n", num_elements);
        printf("Input Array is: ");
        
        for(i = 0; i < num_elements; i++) {
            printf( "%d ", h_input[i]);
        }

        printf( "\n");

        //host arrays
        int *h_output;

        //allocate memory for the host arrays
        //h_input = (int*)malloc(arr_size);
        h_output = (int*)malloc(arr_size);
        
        //device arrays
        int *d_input, *d_output;
        
        //allocate memory for device arrays
        cudaMalloc((void **) &d_input, arr_size);
        cudaMalloc((void **) &d_output, arr_size);
        
        cudaMemcpy(d_input, h_input, arr_size, cudaMemcpyHostToDevice);

        CPU_output(h_input, num_elements);
        
        GPU_output <<< nblocks, nthreads >>>(d_input, d_output);
        cudaMemcpy(h_output, d_output, arr_size, cudaMemcpyDeviceToHost);
        
        printf( "Output  array: GPU : ");
        for(i = 0; i < num_elements; i++) {
            printf("%d ", h_output[i]);
        }
        printf( "\n");
        
        cudaFree(d_input);
        cudaFree(d_output);

        return 0;
    } //END ELSE
}//END MAIN

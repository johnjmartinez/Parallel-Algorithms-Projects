#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define DATASIZE    1000000
#define BLOCK_SIZE  512

void printArray(int a[],int n) {
    FILE *fptr;
    fptr = fopen("Brick_sort_result.txt", "w");

    for(int i=0; i < n; i++) {
        fprintf(fptr,"%d ",a[i]);

	if((i+1)%10 == 0)
	    fprintf(fptr,"\n");
    }

    fclose(fptr);
}

__global__ void oddevensort ( int * input, unsigned int size, int i ) {

    unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;

    int temp;

    if(myId > size)
	return;

    if( i == 0 ) {
	//For even threads
	if(( myId % 2 == 0 && input[myId] > input[myId+1])) {
	    temp = input[myId+1];
	    input[myId+1] = input[myId];
	    input[myId] = temp;
	}
    }
    else {
	//for odd threads
	if(( myId % 2 != 0 && input[myId] > input[myId+1])) {
	    temp = input[myId+1];
	    input[myId+1] = input[myId];
	    input[myId] = temp;
	}
    }

    __syncthreads();
}


int main() {

    float et=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    unsigned int arr_size = DATASIZE * sizeof(int);
    
    //host arrays
    int h_input[DATASIZE], h_output[DATASIZE];
    
    //device arrays
    int *d_input, *d_output;
    
    //allocating memory for device arrays
    cudaMalloc((void**)&d_input, arr_size );
    cudaMalloc((void**)&d_output, arr_size );
    
    srand(time(NULL));
    
    //generating host array values
    for( int i = 0; i < DATASIZE; i++ )
	h_input[i] = rand()%999;
    
    printf("Input array size : %d:\n",DATASIZE);
    if(DATASIZE<=100) {
	for( int i = 0; i < DATASIZE; i++ )
	    printf("%d ", h_input[i] );
    }
    printf("\n");
    
    
    //copy from host to device
    cudaMemcpy( d_input, h_input, arr_size, cudaMemcpyHostToDevice);
    
    int nthreads( BLOCK_SIZE );
    int nblocks( ceil((DATASIZE-1)/(float)BLOCK_SIZE) + 1 );
    
    cudaEventRecord(start);
    
    for( int i=0; i<DATASIZE; i++)
	oddevensort<<< nblocks,nthreads >>>( d_input, DATASIZE, i%2 );
    
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    printf("Time is: %f\n",et);
    
    cudaMemcpy( h_output, d_input, arr_size, cudaMemcpyDeviceToHost);
    
    printArray(h_output,DATASIZE);
    
    printf("\n");
    
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define DATASIZE  1000000
#define BLOCK_SIZE 128

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

__global__ void oddevensort ( int * input, unsigned int size, int i )  {
    
    unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;

    int temp;
    int p;

    if(myId > size) return;

    if( i == 0 ) {
        p=myId*2;   // For even threads  
        // if(( myId % 2 == 0 && input[myId] > input[myId+1]))
        if(input[p]>input[p+1]) {
            temp = input[p+1];
            input[p+1] = input[p];
            input[p] = temp;
        }
    }
    else {
        p=myId*2+1; // for odd threads   
        // if(( myId % 2 != 0 && input[myId] > input[myId+1]))
        if(p<size-1){
            if(input[p]>input[p+1]) {
                temp = input[p+1];
                input[p+1] = input[p];
                input[p] = temp;
            }
        }
    }
    __syncthreads();
}

int main()  {
    
    float et=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    unsigned int arr_size = DATASIZE * sizeof(int);

    int h_input[DATASIZE], h_output[DATASIZE]; // Host []s
    int *d_input, *d_output; // Dev []s

    cudaMalloc((void**)&d_input, arr_size ); // allocating memory for Dev[]s
    cudaMalloc((void**)&d_output, arr_size );

    srand(time(NULL));

    for( int i = 0; i < DATASIZE; i++ ) h_input[i] = rand(); // generating Host [] values
    printf("Input array size : %d:\n",DATASIZE);
    
    if(DATASIZE<=100) {
        for( int i = 0; i < DATASIZE; i++ ) printf("%d ", h_input[i] );
    }
    printf("\n");

    int nthreads( BLOCK_SIZE );
    int nblocks( ceil((DATASIZE-1)/(float)BLOCK_SIZE) + 1 );

    cudaEventRecord(start);     // start time

    cudaMemcpy( d_input, h_input, arr_size, cudaMemcpyHostToDevice); // copy from host to device

    for( int i=0; i<DATASIZE; i++) 
        oddevensort<<< nblocks,nthreads >>>( d_input, DATASIZE, i%2 );

    cudaMemcpy( h_output, d_input, arr_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);     // end time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    printf("Time is: %f\n",et);


    for(int i=0; i<DATASIZE-1; i++) { //  CHECK
        if(h_output[i]>h_output[i+1]) {
            printf("Sorting Failed!!");
            break;
        }
        else if(i == DATASIZE-2)
            printf("Sorting success!");
    }

    printArray(h_output,DATASIZE);
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

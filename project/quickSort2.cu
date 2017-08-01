#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

void printArray(int *a, int len, const char *fileName) {
    FILE *fptr;
    fptr = fopen(fileName, "w");
    for (int i = 0; i < len; i++) fprintf(fptr, "%d\n", a[i]);
    fclose(fptr);
}

__device__ int d_size;

//swap function
__device__ void swap(int *x,int *y) { 
    int temp = *x;
    *x = *y;
    *y = temp;
}

__global__ void partition (int *arr, int *arr_l, int *arr_h, int n) {
    
    int myId = blockIdx.x * blockDim.x + threadIdx.x;
    d_size = 0;

    __syncthreads();

    if(myId >=n) return;

    if (myId < n) {
        int end = arr_h[myId];
        int begin = arr_l[myId];

        int pivot = arr[end];
        int p_index = (begin - 1);

        for (int i = begin; i <= end- 1; i++) {
            if (arr[i] <= pivot) {
                p_index++;
                swap( &arr[i], &arr[p_index] );
            }
        }

        swap( &arr[p_index+1], &arr[end]);

        int j = (p_index + 1);
        if (j-1 > begin) {
            int index = atomicAdd(&d_size, 1);
            arr_l[index] = begin;
            arr_h[index] = j-1;
        }
        if ( j+1 < end ) {
            int index = atomicAdd(&d_size, 1);
            arr_l[index] = j+1;
            arr_h[index] = end;
        }
    }
}

//quicksort
float quickSort (int h_input[], int begin, int end) {

    int nthreads;
    int nblocks;

    float et=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int h_low[end - begin + 1], h_high[end - begin + 1];  //pointers

    int top = -1, *d_input, *d_low, *d_high;

    h_low[ ++top ] = begin;
    h_high[ top ] = end;

    int arr_size = (end+1)*sizeof(int);
    int mysize;
    int k=1;

    cudaMalloc((void **) &d_input, arr_size);
    cudaMemcpy(d_input, h_input,arr_size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_low, arr_size);
    cudaMemcpy(d_low, h_low, arr_size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_high, arr_size);
    cudaMemcpy(d_high, h_high, arr_size, cudaMemcpyHostToDevice);

    nthreads = 1;
    nblocks = 1;

    cudaEventRecord(start);
    while ( k > 0 ) {
        partition<<< nblocks, nthreads >>>( d_input, d_low, d_high, k );
        cudaMemcpyFromSymbol(&mysize, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost);

        if (mysize < 1024) 
            nthreads = mysize;
        else {
            nthreads = 1024;
            nblocks = mysize/nthreads + (mysize%nthreads == 0 ? 0 : 1);
        }
        
        k = mysize;
        //cudaEventRecord(stop);
        //cudaEventSynchronize(stop);    
    }
    
    cudaMemcpy(h_input, d_input,arr_size,cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    return et;
}

int main(int argc, char **argv) {

    int DATASIZE = atoi(argv[1]); 
    int h_input[DATASIZE];
    srand(time(NULL));

    //allocating host array with random values
    for (int i = 0; i<DATASIZE; i++) h_input[i] = rand ();

    float et = quickSort(h_input, 0, DATASIZE-1);
    printf("%d Elements, Time : %.3lf ms\n", DATASIZE, et);

    
    //testing sort
    for(int i=0;i<DATASIZE-1;i++) {
        if (h_input[i]>h_input[i+1]) {
            printf("Sorting Failed\n");
            break;
        }
        else if(i== DATASIZE-2) printf("Sorting Successful\n");
    }
    
    //printing result printArray( h_input, DATASIZE, "output");
    return 0;
}

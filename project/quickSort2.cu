#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_common.h>     

#define LOOPS 10

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
void quickSort (int h_data[], int begin, int end) {

    float et = 0;
    float tmp_time = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int INT_SIZE = sizeof(int);
    int h_low[end - begin], h_high[end - begin];  //pointers
    int top = -1, *d_input, *d_low, *d_high;
    int arrAlloc = end*INT_SIZE;

    h_low[ ++top ] = begin;
    h_high[ top ] = end;

    cudaMalloc((void **) &d_input, arrAlloc);
    cudaMemcpy(d_input, h_data,arrAlloc, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_low, arrAlloc);
    cudaMemcpy(d_low, h_low, arrAlloc, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_high, arrAlloc);
    cudaMemcpy(d_high, h_high, arrAlloc, cudaMemcpyHostToDevice);

    checkCudaErrors(cudaDeviceSynchronize());
	for (int i = 0; i < LOOPS ; i++) {
    
        int mysize;
        int k = 1;
        int nthreads = 1;
        int nblocks = 1;

        checkCudaErrors(cudaEventRecord(start)); //start tmp_time
        while ( k > 0 ) {
            partition<<< nblocks, nthreads >>>( d_input, d_low, d_high, k );
            cudaMemcpyFromSymbol(&mysize, d_size, INT_SIZE, 0, cudaMemcpyDeviceToHost);

            if (mysize < 1024) 
                nthreads = mysize;
            else {
                nthreads = 1024;
                nblocks = mysize/nthreads + (mysize%nthreads == 0 ? 0 : 1);
            }

            k = mysize;
        }
        checkCudaErrors(cudaEventRecord(stop)); // end tmp_time
        checkCudaErrors(cudaEventSynchronize(stop));
        
        tmp_time = 0;
        checkCudaErrors(cudaEventElapsedTime(&tmp_time, start, stop));
        et += tmp_time;
	}
    checkCudaErrors(cudaDeviceSynchronize());    
    
    tmp_time = et/1000/LOOPS;
    printf("Throughput =%9.3lf MElements/s, Time = %.3lf ms\n", 
        1e-6 * end / tmp_time, tmp_time * 1000);
 
    cudaMemcpy(h_data, d_input, arrAlloc, cudaMemcpyDeviceToHost);
}

int main(int argc, char **argv) {
    
    cudaDeviceReset();
    DisplayCudaDevice();

    int DATASIZE = atoi(argv[1])+1; 
    int h_data[DATASIZE];
    
    srand(time(NULL));
    for (int i = 0; i<DATASIZE; i++) h_data[i] = rand ();
    printf("Sorting %d elements\n", DATASIZE); 
    
    quickSort(h_data, 0, DATASIZE);
    
    for(int i=0;i<DATASIZE-1;i++) {
        if (h_data[i]>h_data[i+1]) {
            printf("Sorting Failed\n");
            break;
        }
        else if(i== DATASIZE-2) printf("Sorting Successful\n");
    }    
    //printArray( h_data, DATASIZE, "output");
    return 0;
}

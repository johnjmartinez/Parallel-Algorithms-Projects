
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
 
#define nthreads 450
#define nblocks 1

//Step 1 - split the bits and create an intermediate array of all the bits in 0th, 1th ,2nd--9th position 
//Step 2 - Prefix sum of the intermediate array
//Step 3 - Move the element at the desired location depending on its bit value.

//this function performs the prefix scan of the bit array which is then used to determine the actual position where the element should be positioned in the array.
__device__ int prefix_scan(int *a)
{
   // unsigned int myId = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;  //  # of threads per block

    unsigned int offset;

    for( offset = 1; offset < block_size; offset *= 2) {
        int i;

        if ( tid >= offset )
            i = a[tid-offset];

        __syncthreads();

        if ( tid >= offset )
            a[tid] = i + a[tid];

        __syncthreads();
    }
    return a[tid];
}

//this function extracts the individual bit, and repositions the element to the correct position depending on the bit value 
__device__ void split(int *d_input, int *d_output,unsigned int bit)
{
    //unsigned int myId = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    unsigned int value_tid = d_input[tid];          // value of integer at position tid
    unsigned int bit_tid = (value_tid >> bit) & 1;   // value of bit at position bit, logical bit & to get the correct bit.

    d_input[tid] = bit_tid;        //for the 0th iteration,d_input array contains all bits in 0 position 

    __syncthreads();               // wait until all the threads fill in their bit 

    unsigned int nbits = prefix_scan(d_input);  // prefix scan of all 0's and 1's
    
    unsigned int nbits_1  = d_input[block_size-1];  //total number of 1 bits 
    
    unsigned int nbits_0  = block_size - nbits_1;   //total number of 0 bits
    
    __syncthreads();

    //position the element to the correct location depending on the bit value. 
    if ( bit_tid == 0 )
            d_input[tid - nbits] = value_tid;


    if ( bit_tid == 1 )
        d_input[nbits-1 + nbits_0] = value_tid;
 }

// Kernel function radix_sort
__global__ void GPU_radix_sort(int *d_input,int *d_output)
{
    int  bit;
    for( bit = 0; bit < 10; ++bit )          // max number is 999 and this can be represented using 10 bits, so the for loop will iterate from bit =0 to bit =9.
    {
        split(d_input, d_output,bit);        // eg(d_input,0)
        __syncthreads();
    }
}


/*
__global__ void my_test(int *values)
{
    values[threadIdx.x] +=1;
        __syncthreads();

}
*/


//sequential sort CPU function
 //Step 1 - first find the max from the input array
 //Step 2 - Extract every digit and sort the elements into resp buckets for each digit iteration.
void CPU_radix_sort(int *a, int n) {

     //int bucket[10];
     int divisor = 1;
	 int D[1000];
	 int max = a[0],i;

     //finds the max number from the array
	for (i = 0; i < n; i++) {
		if (a[i] > max)
		   max = a[i];
	}
    // printf("Max number is : %d ", max);

   //Sorts the numbers into buckets based on LSD
	while (max / divisor > 0) {

		int bucket[10] = {0};

		for (i = 0; i < n; i++)
		{
		   int digit = (a[i] / divisor % 10); //putting all the numbers in the respective buckets based on on LSD-MSD
		   bucket[digit]++;
        }
		for (i = 1; i < 10; i++)
		   bucket[i] = bucket[i] + bucket[i - 1];

		for (i = n - 1; i >= 0; i--)
		{

		      D[--bucket[a[i] / divisor % 10]] = a[i];
        }

		for (i = 0; i < n; i++)
		   a[i] = D[i];

		divisor = divisor*10;         // now the LSD digit is done, moving to the next digit.

	}

	printf("Sorted array using CPU(Sequential) sort: \n");
	for (i = 0; i < n; i++)
	{
	printf("%d ",a[i]);
	}

	printf("\n");

}


//-----------------------------------------------

int main(int argc, char *argv[])
{

// here I am giving a command line arg - like sort.cu file_50.txt 50 where 50 is the number of records in the file.
    if ( argc != 3 ) /* argc should be 3 for correct execution */
    {
        /* We print argv[0] assuming it is the program name */
        printf( "usage: %s file1", argv[0] );
        return 0;
    }
    else
    {
        int num_elements = atoi(argv[2]);
        const int N = nthreads*nblocks;
        const int arr_size = N*sizeof(int);

        int *h_input = (int *)malloc(arr_size);    //host input array
        int i;

        FILE * pFile1;

        pFile1 = fopen (argv[1],"r");

        if (pFile1 == 0)
        {
            printf("could not open file\n");
            printf(" Error %d",errno );
            // return 0;
        }


        if(pFile1!=NULL) {
            for (i = 0; i < num_elements; i++)
            {
            fscanf(pFile1, "%d,", &(h_input[i]));
            }
        }

        printf("Number of elements in the array : %d \n", num_elements);

        printf( "Input Array is: ");
        for(i = 0; i < num_elements; i++)
        {
           printf( "%d ", h_input[i]);
        }
 
      printf("\n");   
      fclose(pFile1);

//host output array
int *h_output;

//allocate memory for the host arrays
h_output = (int*)malloc(arr_size);

//device arrays
int *d_input,*d_output;

//allocate memory for device arrays
cudaMalloc((void **) &d_input, arr_size);
cudaMalloc((void **) &d_output, arr_size);

cudaMemcpy(d_input,h_input,arr_size,cudaMemcpyHostToDevice);
//cudaMemcpy(d_output,h_input,arr_size,cudaMemcpyHostToDevice);

//int min = minimum(h_input,num_elements);

CPU_radix_sort(h_input,num_elements);

GPU_radix_sort<<< nblocks,nthreads >>>(d_input,d_output);

cudaMemcpy(h_output,d_input,arr_size,cudaMemcpyDeviceToHost);

//printf(" Sequential CPU: Min number is %d ",min );
printf("Sorted array using GPU(Parallel) sort: \n ");
for(i=0;i<num_elements;i++)
{
printf( "%d ",h_output[i] );
}

printf("\n");


cudaFree(d_input);
cudaFree(d_output);

return 0;
}
}

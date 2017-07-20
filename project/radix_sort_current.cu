
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

//#define nthreads 512
//#define nblocks 200
#define DATASIZE    600
#define BLOCK_SIZE   512

//Step 1 - split the bits and create an intermediate array of all the bits in 0th, 1th ,2nd--9th position 
//Step 2 - Prefix sum of the intermediate array
//Step 3 - Move the element at the desired location depending on its bit value.

//this function performs the prefix scan of the bit array which is then used to determine the actual position where the element should be positioned in the array.
__device__ int prefix_scan(int *a)
{

 //   unsigned int myId = threadIdx.x;
    unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
    
    unsigned int block_size = blockDim.x;  //  # of threads per block

    unsigned int offset;

    for( offset = 1; offset < block_size; offset *= 2) {
        int i;

        if ( myId >= offset )
            i = a[myId-offset];

        __syncthreads();

        if ( myId >= offset )
            a[myId] = i + a[myId];

        __syncthreads();
    }
    return a[myId];
}

//this function extracts the individual bit, and repositions the element to the correct position depending on the bit value 
__device__ void split(int *d_input, int *d_output,unsigned int bit,int size)
{
   // unsigned int myId = threadIdx.x;
    unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;

    unsigned int block_size = blockDim.x;

    //if(myId>=size)
   // return;

    unsigned int value_tid = d_input[myId];          // value of integer at position tid
    unsigned int bit_tid = (value_tid >> bit) & 1;   // value of bit at position bit, logical bit & to get the correct bit.

    d_input[myId] = bit_tid;        //for the 0th iteration,d_input array contains all bits in 0 position

    __syncthreads();               // wait until all the threads fill in their bit 

    unsigned int nbits = prefix_scan(d_input);  // prefix scan of all 0's and 1's
    
    unsigned int nbits_1  = d_input[block_size-1];  //total number of 1 bits 
    
    unsigned int nbits_0  = block_size - nbits_1;   //total number of 0 bits
    
    __syncthreads();

    //position the element to the correct location depending on the bit value. 
    if ( bit_tid == 0 )
            d_input[myId - nbits] = value_tid;


    if ( bit_tid == 1 )
        d_input[nbits-1 + nbits_0] = value_tid;
}

// Kernel function radix_sort
__global__ void GPU_radix_sort(int *d_input,int *d_output,int size)
{
    int  bit;
    for( bit = 0; bit < 10; ++bit )          // max number is 999 and this can be represented using 10 bits, so the for loop will iterate from bit =0 to bit =9.
    {
        split(d_input, d_output,bit,size);        // eg(d_input,0)
        __syncthreads();
    }
}



//-----------------------------------------------

int main()
{
   float et=0;
   cudaEvent_t start, stop;
   cudaEventCreate(&start); 
   cudaEventCreate(&stop);

// here I am giving a command line arg - like sort.cu file_50.txt 50 where 50 is the number of records in the file.
//    if ( argc != 3 ) /* argc should be 3 for correct execution */
//    {
        /* We print argv[0] assuming it is the program name */
//        printf( "usage: %s file1", argv[0] );
//       return 0;
//    }
/*    else
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
*/
       /* printf( "Input Array is: ");
        for(i = 0; i < num_elements; i++)
        {
           printf( "%d ", h_input[i]);
        }*/
 
  //    printf("\n");   
//      fclose(pFile1);

int h_input[DATASIZE], h_output[DATASIZE];

for( int i = 0; i < DATASIZE; i++ )
{
h_input[i] = rand() % 900;
}

printf("Input array size :%d \n",DATASIZE);
if(DATASIZE <=600)
{
//printf("Input array size : %d:\n",DATASIZE);
for( int i = 0; i < DATASIZE; i++ )
{
printf("%d ", h_input[i] );
}
}
printf("\n");

//host output array
//int *h_output;

unsigned int MemDataSize = DATASIZE * sizeof(int);

//allocate memory for the host arrays
//h_output = (int*)malloc(arr_size);

//device arrays
int *d_input,*d_output;

//allocate memory for device arrays
cudaMalloc((void **) &d_input, MemDataSize);
cudaMalloc((void **) &d_output, MemDataSize);

cudaMemcpy(d_input,h_input,MemDataSize,cudaMemcpyHostToDevice);
//cudaMemcpy(d_output,h_input,arr_size,cudaMemcpyHostToDevice);


int nthreads = 512;
int numBlocks = (DATASIZE + nthreads - 1) / nthreads;

//int    dimBlocksize( BLOCK_SIZE );
int   dimGridsize( ceil((DATASIZE-1)/(float)BLOCK_SIZE) + 1 );

cudaEventRecord(start);
GPU_radix_sort<<< dimGridsize,DATASIZE>>>(d_input,d_output,DATASIZE);
//GPU_radix_sort<<< numBlocks,nthreads >>>(d_input,d_output,DATASIZE);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
//cudaEventSyncronize(stop);
cudaEventElapsedTime(&et, start, stop);

printf("Time is: %f\n",et);

cudaMemcpy(h_output,d_input,MemDataSize,cudaMemcpyDeviceToHost);

printf("Sorted array using GPU(Parallel) sort: \n ");
if(DATASIZE<=600)
{
for(int i=0;i<DATASIZE;i++)
{
printf( "%d ",h_output[i] );
}
}
printf("\n");

cudaFree(d_input);
cudaFree(d_output);

return 0;
}



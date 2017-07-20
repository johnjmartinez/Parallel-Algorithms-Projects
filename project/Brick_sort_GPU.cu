#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define DATASIZE    100000
#define BLOCK_SIZE    512

__global__ void oddevensort ( int * input, unsigned int size, int i )
{

unsigned int tid = threadIdx.x;
//unsigned int myid = tid + blockDim.x * blockIdx.x;

int temp;
int p;

if( i % 2 == 0 )
{
p=tid*2;

if(input[p]>input[p+1])
{
temp = input[p+1];
input[p+1] = input[p];
input[p] = temp;
}
}
else
{
p=tid*2+1;

if(p<size-1){
if(input[p]>input[p+1])
{
temp = input[p+1];
input[p+1] = input[p];
input[p] = temp;
}
}
}
__syncthreads();
}



int main()
{

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

//generating host array values
for( int i = 0; i < DATASIZE; i++ )
{
h_input[i] = rand() % 900;
}


printf("Input array size : %d:\n",DATASIZE);
if(DATASIZE<=100)
{
for( int i = 0; i < DATASIZE; i++ )
{
printf("%d ", h_input[i] );
}
}
printf("\n");

//cudaMemset( d_output, 0, arr_size );
//
//copy from host to device
cudaMemcpy( d_input, h_input, arr_size, cudaMemcpyHostToDevice);

//int    nthreads( BLOCK_SIZE );
int   nblocks( ceil((DATASIZE-1)/(float)BLOCK_SIZE) + 1 );

cudaEventRecord(start);
for( int i=0; i<DATASIZE; i++)
{
oddevensort<<< nblocks,DATASIZE/2 >>>( d_input, DATASIZE, i );
}
cudaEventRecord(stop);

cudaEventSynchronize(stop);
cudaEventElapsedTime(&et, start, stop);
printf("Time is: %f\n",et);

cudaMemcpy( h_output, d_input, arr_size, cudaMemcpyDeviceToHost);
printf("Sorted array:\n");
if(DATASIZE<=100)
{
for( int i=0; i<DATASIZE; i++ )
{
printf("%d ",h_output[i]);
}
}
printf("\n");

cudaFree( d_input );
cudaFree( d_output );
return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>

#define DATASIZE 10000

#define TPB 256

void printResult( int h_input[], int n )
{
FILE *fptr;
fptr = fopen("Qsort.txt","w");
fprintf(fptr,"Number of elements sorted:%d\n ",n);

for (int i = 0; i < n; ++i )
{
fprintf(fptr, "%d ", h_input[i] );

if((i+1)%10 ==0)
fprintf(fptr,"\n");
}
fclose(fptr);
}

typedef struct vars{
int begin;
int end;
int low;
} vars;


__global__ void partition(int * d_input, int * d_output, vars * p,int pivot, int begin, int end,int d_low[],
int d_high[], int *d_low_val, int *d_high_val,int nblocks)
{

//use of shared mem
__shared__ int s_input[TPB];
__syncthreads();

int tid =threadIdx.x;
int myId = begin + blockIdx.x*TPB + tid;

__shared__ int begin_block, end_block;
__shared__ int begin_offset, end_offset;

if(tid == 0){
d_low[blockIdx.x] = 0;
d_high[blockIdx.x] = 0;
*d_low_val = 0;
*d_high_val = 0;

}
__syncthreads();

if(myId <= (end - 1)){
s_input[tid] = d_input[myId];


if( s_input[tid] <= pivot ){
atomicAdd( &(d_low[blockIdx.x]), 1);
} else {
atomicAdd( &(d_high[blockIdx.x]), 1);
}

}
__syncthreads();


if (tid == 0){
begin_block = d_low[blockIdx.x];
begin_offset = begin+atomicAdd(d_low_val, begin_block);
}
if (tid == 1){
end_block = d_high[blockIdx.x];
end_offset = end-atomicAdd(d_high_val, end_block);
}

__syncthreads();

if(tid == 0){


int m = 0;
int n = 0;
for(int j = 0; j < TPB; j++){
int chk = begin + blockIdx.x*TPB + j;
if(chk <= (end-1) ){
if(s_input[j] <= pivot){

d_output[begin_offset + m] = s_input[j];
++m;
} else {

d_output[end_offset - n] = s_input[j];
++n;
}
}
}
}

__syncthreads();

if((blockIdx.x == 0) && (tid == 0)){
int pOffset = begin;
for(int k = 0; k < nblocks; k++)
pOffset += d_low[k];

d_output[pOffset] = pivot;
p->begin = (pOffset - 1);
p->end = (pOffset + 1);
}

return;
}

/********************/

void quickSort(int h_input[], int begin, int end, int n){

if((end - begin) >= 1){

//pivot
int pivot = h_input[end];


int nblocks = (end - begin) / TPB;
if((nblocks * TPB) < (end - begin))
nblocks++;

//device arrays
int * d_input;
int * d_output;
int * d_low, * d_high, *d_low_val, *d_high_val;
vars * d_endpts;

vars p;

p.begin = begin;
p.end = end;

int arr_size = n*sizeof(int);

//allocating device arrays
cudaMalloc(&(d_input), arr_size);
cudaMalloc(&(d_output), arr_size);
cudaMalloc(&(d_endpts), sizeof(vars));

cudaMalloc(&(d_low), 4*nblocks);
cudaMalloc(&(d_high), 4*nblocks);

cudaMalloc(&d_low_val, 4);
cudaMalloc(&d_high_val, 4);

//copy from host to device
cudaMemcpy(d_input, h_input, arr_size, cudaMemcpyHostToDevice);
cudaMemcpy(d_output, h_input, arr_size, cudaMemcpyHostToDevice);

//gpu partition
partition<<<nblocks, TPB>>>(d_input, d_output, d_endpts, pivot, begin, end, d_low, d_high, d_low_val, d_high_val, nblocks);

//copy results back to host
cudaMemcpy(h_input, d_output, arr_size, cudaMemcpyDeviceToHost);
cudaMemcpy(&(p), d_endpts, sizeof(vars), cudaMemcpyDeviceToHost);

cudaThreadSynchronize();

cudaFree(d_input);
cudaFree(d_output);
cudaFree(d_endpts);
cudaFree(d_low);
cudaFree(d_high);

//recursive call to Quicksort
if(p.begin >= begin)
quickSort(h_input, begin, p.begin, n);
if(p.end <= end)
quickSort(h_input, p.end, end, n);

}

return;
}
//-----------------------------------------

int main()
{

float et=0;
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

int h_input[DATASIZE];

srand(time(NULL));

//allocating host array with random values
for (int i = 0; i<DATASIZE; i++)
{
h_input[i] = rand ();
}

cudaEventRecord(start);
//quickSort(h_input,0,DATASIZE-1);
quickSort(h_input,0,DATASIZE-1,DATASIZE);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

cudaEventElapsedTime(&et, start, stop);
printf("Time is: %f\n",et);

//testing sort
for(int i=0;i<DATASIZE-1;i++)
{
if (h_input[i]>h_input[i+1])
{
printf("Sorting Failed!!\n");
break;
}
else if(i== DATASIZE-2)
printf("Sorting successful\n");
}

//printing result
printResult( h_input, DATASIZE );

return 0;
}

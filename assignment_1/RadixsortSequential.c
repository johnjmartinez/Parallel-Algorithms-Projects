
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>

//Finds the max number from the input array.
int maximum(int D[], int n)
{
    int max = D[0],i;
    for(i = 1; i < n; i++)
    {
        if(max < D[i])
            max = D[i];
    }
    return max;
}

void RadixSortSequential(int D[], int n)
{
    int max;
    int i, j, k,ite,num_ite,digit,divisor;
   
    num_ite=0;                                          //this is the number of iterations
    divisor=1;
    int basket[10][10], basket_count[10];
    
    //Step 1 - first find the max from the input array
     max = maximum(D, n);
    //printf("The max element %d\n",max);
    
    //Step 2 - find after how many iterations the array will be sorted
    while(max > 0)
    {
        num_ite++;                             // checking how many iterations needs to be done before the array is sorted.
        max = max/10;
    }
    //printf("number of iterations is %d ",num_ite);

    //Step 3 - for Every iteration,put the element in the array in the resp basket moving from LSD to MSD.The divisor is initially 1 and is multipled by 10 every iteration to get to the correct digit.After all the iterations are complete, the indivdual baskets are already sorted, just merge all the baskets to array D.
    for(ite = 0; ite < num_ite; ite++)
    {
        for(i = 0; i < 10; i++)
        {
           basket_count[i] = 0;
        }
        for(i = 0; i < n; i++)
        {
            digit = (D[i] / divisor) % 10;
            basket[digit][basket_count[digit]] = D[i];
            basket_count[digit] += 1;
        }
        
        i = 0;
        for(k = 0; k < 10; k++)
        {
            for(j = 0; j < basket_count[k]; j++)
            {
                D[i] = basket[k][j];
                i++;
            }
        }
        divisor *= 10;
        
    }
}


//main program
int main(int argc, char *argv[])
{
    // here I am giving a command line arg prg name inp.txt n where n is the number of records in the file.
    if ( argc != 3 ) /* argc should be 3 for correct execution */
    {
        /* We print argv[0] assuming it is the program name */
        printf( "usage: %s file1", argv[0] );
        return 0;
    }
    else
    {
        int num_elements = atoi(argv[2]);
        int *input_arr = (int *)calloc(num_elements, sizeof(int));
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
            //printf("%d\n",num_elements );
            for (i = 0; i < num_elements; i++)
            {
            fscanf(pFile1, "%d,", &(input_arr[i]));
            }
        }
            printf("\nArray before sorting: \n");
            for(i = 0;i< num_elements;i++){
                printf("%d ",input_arr[i] );	//print array
            }
        
        RadixSortSequential(input_arr,i);
        
        printf("\nArray After sorting: \n");
        for(i = 0;i< num_elements;i++){
            printf("%d ",input_arr[i] );	//print array
        
    }
        printf("\n");
    return 0;
  
   
   }
}

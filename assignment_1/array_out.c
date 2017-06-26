
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>

//Finds the min number from the input array.
void Output(int D[], int n)
{
    int i, divisor =1;
    int *output_arr = (int *)calloc(n, sizeof(int));
    for(i = 0; i < n; i++)
    {
        output_arr[i] = (D[i] / divisor) % 10;
    }
    
    printf( "Output Array is: ");
    for(i = 0; i < n; i++)
    {
        printf(  "%d ", output_arr[i]);
    }
    
    printf( "\n");

    
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
        printf( "Input Array is: ");
        for(i = 0; i < num_elements; i++)
        {
            printf( "%d ", input_arr[i]);
        }
        
        printf( "\n");

        
        Output(input_arr,i);
    
        
    }
    return 0;
  
}


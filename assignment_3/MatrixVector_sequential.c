

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>

#define N 100

int main(int argc, char *argv[]) {
    
    if ( argc != 3 ) /* argc should be 3 for correct execution */
    {
        /* We print argv[0] assuming it is the program name */
        printf( "usage: %s file1, file2, noOfThreads", argv[0] );
    }
    else
    {
        doMult(argv[1], argv[2]);
    }
    return 0;
}



int doMult(char file1[], char file2[])
{
    
    FILE * pFile1;
    FILE * pFile2;
    
    int R1;
    int C1,C2,sum;
    int a,b,c,d;
    
    //double Matrix1[N][N];
    //double Vector1[N];
    //double Multiply[N];
    
    int Matrix1[N][N];
    int Vector1[N];
    int Result[N];
    
    
    pFile1 = fopen (file1,"r");
    if (pFile1 == 0)
    {
        printf("could not open file\n");
        printf(" Error %d",errno );
        return 0;
    }
    if (pFile1!=NULL) {
        fscanf(pFile1, "%d", &R1);
        fscanf(pFile1, "%d\n", &C1);
        printf("Rows %d, Columns %d", R1,C1);
        for (a = 0; a < R1; a++)
        {
            for (b = 0; b < C1; b++)
            {
               //fscanf(pFile1,"%lf", &Matrix1[a][b]);
                fscanf(pFile1,"%d", &Matrix1[a][b]);

            }
        }
    }
    fclose (pFile1);
    printf("\n");
  //---------------------------
    printf("Printing Matrix: \n");
    for (a = 0; a < R1; a++)
    {
        for (b = 0; b < C1; b++)
        {
            //printf("%lf ", Matrix1[a][b]);
            printf("%d ", Matrix1[a][b]);
        }
         printf("\n");
    }
      printf("\n");
  //---------------------------
   
    
    pFile2 = fopen (file2,"r");
    if (pFile2 == 0)
    {
        printf("could not open file 2\n");
        return 0;
    }
    
    if (pFile2!=NULL)
    {
        
        for (c = 0; c < C1; c++)
        {
           //fscanf(pFile2,"%lf", &Vector1[c]);
            fscanf(pFile2,"%d", &Vector1[c]);
            
        }
    }
    
    fclose (pFile2);
    
    //---------------------------
    printf("Printing Vector: \n");
    for (a = 0; a < C1; a++)
    {
         //printf("%lf ", Vector1[a]);
        printf("%d ", Vector1[a]);

    }
     printf("\n");
    //---------------------------
    
       for(a=0; a < R1; a++)
        {
            int sum =0;
            for(b=0;b < C1; b++)
            {
                sum = sum + Matrix1[a][b] * Vector1[b] ;
            }
                Result[a] = sum;
            
       }
            
    
    FILE *fptr;
    fptr = fopen("Result.txt", "w");
     
    for(a=0; a < R1; a++)
    {
        fprintf(fptr,"%d\t", Result[a]);
          //printf("%d\t", Result[a][0]);
        
        fprintf(fptr,"\n");
        //printf("\n");
    }
    
    printf("Multiplication Results are in Result.txt\n");
    
    fclose(fptr);
    
    return 0;
}





/* HW3 - by John M and Pankaja A. */
/*Implementing matrix vector multiplication in parallel using MPI*/

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "mpi.h"
#include "errno.h"

#define N 100

int AROW,ACOL;
int Matrix1[N][N];
int Vector1[N];
int Result[N];

void Readmatrix();

void printResult();

int row_divide(int i, int size,int R1);


int main(int argc, char** argv)
{
    
    MPI_Status Stat;
    int size, rank;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0)  //master process
    {
        //Reading the input matrix
        if ( argc != 3 ) /* argc should be 3 for correct execution */
        {
            /* We print argv[0] assuming it is the program name */
            printf( "usage: %s file1, file2", argv[0] );
        }
        else
            Readmatrix(argv[1], argv[2]);
    }
    //Broadcasting matrix dimensions.
    MPI_Bcast ( (void *)&ACOL,1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast ( (void *)&AROW,1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) //master process
    {
        //sending vector to other processes
        for (int i=1;i<size;i++)
        {
            MPI_Send(Vector1, ACOL, MPI_INT, i, 99, MPI_COMM_WORLD);
            //printf("Sending vector of size %d to process %d\n",ACOL,i);

        }
        
        //MPI_Bcast(&Vector1,1,MPI_INT, 0, MPI_COMM_WORLD);
       
      //sending part of matrix to other processes
        for (int j=0;j<AROW;j++)
        {
            int proc = row_divide(j, size,AROW);
            //printf("send row %d to process %d\n", j,proc);
            MPI_Send(Matrix1[j], ACOL, MPI_INT, proc, (100*(j+1)), MPI_COMM_WORLD);
        }
        
        //Collecting result from slave processes/workers
        printf("Multiplication Results are in Result.txt\n");
        for (int k=0;k<AROW;k++)
        {
            int worker = row_divide(k, size,AROW);
            MPI_Recv(&Result[k], 1, MPI_INT, worker, k, MPI_COMM_WORLD, &Stat);
            printResult();
        }
        
    }
    
    else  //slave process
    {

        int Vector1[ACOL];
        
        // printf("COLUMN size if Rank is non zer0 %d\n",ACOL);
        // printf("ROW size if Rank is non zer0 %d\n",AROW);
        
        //Receiving vector from master
        MPI_Recv(Vector1, ACOL, MPI_INT, 0, 99, MPI_COMM_WORLD, &Stat);
        
        //actual work
        for (int i=0;i<AROW;i++)
        {
            int pid = row_divide(i, size,AROW);
            if (rank == pid)
            {
                int temp[ACOL];
                MPI_Recv(temp, ACOL, MPI_INT, 0, (100*(i+1)), MPI_COMM_WORLD, &Stat);
                int sum = 0;
                for (int j=0;j<ACOL;j++)
                {
                    sum = sum + (temp[j] * Vector1[j] );
                }
                MPI_Send(&sum, 1, MPI_INT, 0, i, MPI_COMM_WORLD);
            }
        }
    }
    
    MPI_Finalize();
    return 0;
    
    
}

//Dividing the rows amongst n processors.
int row_divide(int i, int size,int R1)
{
    // printf("Size inside row_divide  %d\n",size);
    size = size - 1;
    int r_div = (int) ceil( (double)R1 / (double)size);
    //printf("r_div inside row_divide  %d\n",r_div);
    int pid = i / r_div;
    //printf("pid inside row_divide  %d\n",pid);
    return pid+1;
}


//Reading the Matrix and Vector.
void  Readmatrix(char file1[], char file2[])
    {
        FILE * pFile1;
        FILE * pFile2;
        

        pFile1 = fopen (file1,"r");
        if (pFile1 == 0)
        {
            printf("could not open file\n");
            printf(" Error %d",errno );
            return;
        }
        
        if (pFile1!=NULL) {
            fscanf(pFile1, "%d", &AROW);
            fscanf(pFile1, "%d\n", &ACOL);
           //printf("Rows %d, Columns %d\n", AROW,ACOL);
            for (int i = 0; i < AROW; i++)
            {
                for (int j = 0; j < ACOL; j++)
                {
                    //fscanf(pFile1,"%lf", &Matrix1[a][b]);
                    fscanf(pFile1,"%d", &Matrix1[i][j]);
                    
                }
            }
        }
        fclose (pFile1);
        
        /*printf("\nPrinting Matrix :\n");
        for (int i=0;i<AROW;i++)
        {
            for (int j=0;j<ACOL;j++)
            {
                printf("%d ", Matrix1[i][j]);
            }
            printf("\n");
        }*/
        
        
        pFile2 = fopen (file2,"r");
        if (pFile2 == 0)
        {
            printf("could not open file 2\n");
            return;
        }
        
        //Assuming that the vector has the same number of rows as columns of matrix.
        if (pFile2!=NULL)
        {
            
            for (int i = 0; i < ACOL; i++)
            {
                //fscanf(pFile2,"%lf", &Vector1[c]);
                fscanf(pFile2,"%d", &Vector1[i]);
                
            }
        }
        
        fclose (pFile2);
        
        
        /*printf("\nPrinting Vector :\n");
        for (int i=0;i<ACOL;i++)
        {
            printf("%d ", Vector1[i]);
        }
        printf("\n");*/
        
    }

//Printing result in Result.txt
void printResult()
{
    FILE *fptr;
    fptr = fopen("Result.txt", "w");
    
    for(int a=0; a < AROW; a++)
    {
        fprintf(fptr,"%d\t", Result[a]);
        //printf("%d\t", Result[a]);
        
        fprintf(fptr,"\n");
        //printf("\n");
    }
    
    fclose(fptr);
}



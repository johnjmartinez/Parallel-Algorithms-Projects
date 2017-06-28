#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstdlib>
using namespace std;

// output contents of array to screen
void printArray(int arr[], int size) {
    for ( int i = 0; i < size; i++ ) {
        cout << arr[i] << '\t';
    }
    cout << endl;
}

void countRanges (int count, int *A, int *B) {
    int x;
    for(int i=0; i < count ; i++) {
        x = A[i] / 100;
        B[x]++;
    }
}

void scan (int size, int *B, int *C) {
    for(int i=0; i < size ; i++) {
        if (i==0) C[i] = B[i];
        else C[i] = B[i] + C[i-1];
    }
}

/*
-- Counting nums in specific ranges
2. (40 pts) Read an array A as in the first question.
    (a, 10 pts) Create an array B of size 10 that keeps a count of the entries in each of the ranges: [0,99], [100,199], [200,299], ... Maintain an array B in global memory of GPU.
    
    (b, 10 pts) Repeat part (a) but first use the shared memory in a block for updating the local copy of B in each block. Once every block is done, add all local copies to get the global copy of B.
    
    (c, 20 pts) Create an array of size 10 that uses B to compute C which keeps count of the entries in each of the ranges: [0,99], [0,199], [0,299], ... , [0,999]. Note that the ranges are different from part (a). DO NOT use array A.
*/    
int main(int argc, char *argv[]) {

    // <program> <file> <num of elems in file> -- argc should be 3 for correct execution 
    if ( argc != 3 ) { 
        printf( "usage: %s <file> <num of elem in file>\n", argv[0] );
        return -1;
    }
    
    int count = atoi(argv[2]);
    string tmp, delim = ",";
    size_t last = 0, next; 
   
    ifstream InFile(argv[1]);
    stringstream buffer;
    buffer << InFile.rdbuf();
    tmp = buffer.str();
    //cout << tmp << "\n";  // DEBUG

    int *A = new int[count];
    int *B = new int[10];  
    int *C = new int[10];  
    for(int i=0; i < 10 ; i++) {  
        B[i] = 0;
        C[i] = 0;
    }
      
    // READ FILE INTO A[]
    int x = 0;
    while ( (next = tmp.find(delim, last)) != string::npos) { // this was a pain in the butt ... why not separate using space???
        A[x] = stoi( tmp.substr(last, next-last) );
        last = next + 1; 
        x++;
    } 
    A[x] = stoi( tmp.substr(last) ); // compile using g++ -std=c++11 
    //printArray(A,count); // DEBUG
    
    countRanges(count, A, B);
    printArray(B, 10); // DEBUG
    
    scan(10, B, C);
    printArray(C, 10); // DEBUG
    
    return 0;
  
} //END MAIN






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
        return 0;
    }
    
    int count = atoi(argv[2]), x = 0;
    int A[count], B[10] = {0}, C[10] = {0};
    string tmp, delim = ",";
    size_t last = 0, next; 
    
    ifstream InFile(argv[1]);
    stringstream buffer;
    buffer << InFile.rdbuf();
    tmp = buffer.str();
    //cout << tmp << "\n";  // DEBUG

    
    // READ FILE INTO A[]
    while ( (next = tmp.find(delim, last)) != string::npos) { // this was a pain in the butt ... why not separate using space???
        A[x] = stoi( tmp.substr(last, next-last) );
        last = next+1; 
        x++;
    } 
    A[x] = stoi( tmp.substr(last) ); // compile using g++ -std=c++11 
    
    // COUNT RANGES
    for(int i=0; i < count ; i++) {
        x = A[i] / 100;                   // POSITIVE TRUNCATION
        cout << A[i] << ":" << x << " ";  // DEBUG

        B[x]++;
        for (int j=9 ; j >= x ; j--)  C[j]++;    
        
    /* BRUTE FORCE APPROACH
        if (A[i] < 100) {      
            B[0]++; C[9]++; C[8]++; C[7]++; C[6]++; C[5]++; C[4]++; C[3]++; C[2]++; C[1]++; C[0]++;
        }
        else if (A[i] < 200) { 
            B[1]++; C[9]++; C[8]++; C[7]++; C[6]++; C[5]++; C[4]++; C[3]++; C[2]++; C[1]++; 
        }
        else if (A[i] < 300) { 
            B[2]++; C[9]++; C[8]++; C[7]++; C[6]++; C[5]++; C[4]++; C[3]++; C[2]++; 
        }
        else if (A[i] < 400) { 
            B[3]++; C[9]++; C[8]++; C[7]++; C[6]++; C[5]++; C[4]++; C[3]++; 
        }
        else if (A[i] < 500) { 
            B[4]++; C[9]++; C[8]++; C[7]++; C[6]++; C[5]++; C[4]++; 
        }
        else if (A[i] < 600) { 
            B[5]++; C[9]++; C[8]++; C[7]++; C[6]++; C[5]++; 
        }
        else if (A[i] < 700) { 
            B[6]++; C[9]++; C[8]++; C[7]++; C[6]++; 
        }
        else if (A[i] < 800) { 
            B[7]++; C[9]++; C[8]++; C[7]++; 
        }
        else if (A[i] < 900) { 
            B[8]++; C[9]++; C[8]++; 
        }
        else if (A[i] < 1000) { 
            B[9]++; C[9]++; 
        }
    */            
    } //END FOR COUNTS B[], C[]
    cout << "\n";

    //printArray(A,count); // DEBUG
    printArray(B,10); // DEBUG
    printArray(C,10); // DEBUG
    
    return 0;
  
} //END MAIN






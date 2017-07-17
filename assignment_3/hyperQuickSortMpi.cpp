#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <vector>
#include <mpi.h>

//#define debug
using namespace std;
/*
    PARALLEL HYPERQUICKSORT
    Read an array A from a file named input.txt.  
    Divide the input array into roughly equal sized chunks n/p chunks (total rows: n, total processes: p) 
    Compute an array D such that it is sorted in ascending order using Parallel HyperQuickSort. 
    After HyperQuickSort array D is split among p processors such that the higher numbered processor has higher entries. 
    Write the output in a file called output.txt
*/     




int num_procs, id; // MAKE NUM_PROC & IDS GLOBAL

// MAIN
int main(int argc, char *argv[]) {

    vector<int> *chunkVect;
    int chunkSize;

    MPI::Init( &argc, &argv );

    num_procs = MPI::Comm::Get_size(); // Get processors number.
    id = MPI::Comm::Get_rank(); // Get curr processor id/rank.

    if ( argc != 2 ) {
        cout <<" Usage: %s <input.txt>" << argv[0] << endl;
        return -1;
    }
    
    //READ ARRAY AND SCATTER EVENLY AMONG PROCS
    if ( id == 0 ) {

        ifstream inFile( argv[1]) ;
        istream_iterator<int> start(inFile), end;
        vector<int> A(start, end);
    
        cout << "Read " << A.size() << " numbers" << endl;

#ifdef debug
#include <algorithm> 
        cout << "numbers read in:\n";
        copy(A.begin(), A.end(), ostream_iterator<int>(cout, " "));
        cout << endl;
#endif    

        chunkSize = A.size()/num_procs;
    }
    
    MPI::Comm::Bcast( &chunkSize, 1, MPI_INT, 0, MPI_COMM_WORLD); // only root can Bcast (id=0)
    *chunkVect(chunkSize, 0); // init vector size

    //https://www.open-mpi.org/doc/v1.10/man3/MPI_Scatter.3.php    
    MPI::Comm::Scatter( &A[(id)*(chunkSize)], chunkSize, MPI_INT, // sending info -- only root can scatter
                        chunkVect, chunkSize, MPI_INT,            // where to receive it
                        0, MPI_COMM_WORLD)/

    //hyperQuickSort(A, 0, A.size()-1);
   
    for(auto& x:A)
         cout << ' ' << x;
    cout << endl;
    return 0;
  
   
}

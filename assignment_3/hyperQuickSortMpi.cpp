#include <algorithm> 
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include <mpi.h>

#define ROOT 0 
#define LOWER_BOUND -99999 // based on python script that generates input.txt
#define UPPER_BOUND  99999

#define debug2
using namespace std;
/*
    PARALLEL HYPERQUICKSORT (HQsort)
    Read an array A from a file named input.txt.  
    Divide the input array into roughly equal sized chunks n/p chunks (total rows: n, total processes: p) 
    Compute an array D such that it is sorted in ascending order using Parallel HyperQuickSort. 
    After HyperQuickSort array D is split among p procs s.t. higher numbered proc has higher entries. 
    Write the output in a file called output.txt
    
    ALGORITHM:
    for i = d down to 1
        for each i-cube:
            root of the i-cube broadcasts its median to all in the i-cube, to serve as pivot
            consider the two (i-1)-subcubes of this i-cube
            each pair of partners in the (i-1)-subcubes exchanges data:
                low-numbered PE gives its partner its data larger than pivot
                high-numbered PE gives its partner its data smaller than pivot
*/     

int getMedian( vector<int>& V ) {
    const auto iter1 = V.begin() + V.size() / 2 - 1;
    const auto iter2 = V.begin() + V.size() / 2;
    nth_element(V.begin(), iter1 , V.end()); 
    nth_element(V.begin(), iter2 , V.end()); 
    auto median = (V.size() % 2 == 0) ? (*iter1 + *iter2) / 2 : *iter2;
    return median;
}

void HQsort (vector<int>& V, int& size) {
    int pivot, incoming_size;
    vector<int> incoming_V;

    int num_procs = MPI::COMM_WORLD.Get_size(); // Get number of procs.
    int curr_id = MPI::COMM_WORLD.Get_rank(); // Get curr processor id/rank.
 
    sort(V.begin(), V.end()); //locally sort proc vectors
    
    if ( curr_id == ROOT ) {
        pivot = getMedian(V);
        cout << "Pivot " << pivot << endl;
    }
    
    MPI::COMM_WORLD.Bcast( &pivot, 1, MPI::INT, ROOT ); // Bcast pivot
    
    const auto iter = stable_partition( V.begin(), V.end(), [pivot](int n){ return n < pivot; } ); // ninja lambda
    vector<int> lower(V.begin(), iter); // less than pivot 
    vector<int> upper(iter, V.end());  // equal to or more than pivot
    
    int partner_id = ( curr_id + (num_procs/2) ) % num_procs;

    if( curr_id < partner_id ) { 
        // https://www.open-mpi.org/doc/v1.10/man3/MPI_SendrecV.3.php
        // exchange sizes first
        int chunk_size = upper.size();
        MPI::COMM_WORLD.Sendrecv(&chunk_size, 1, MPI::INT, partner_id, 0,
                                 &incoming_size, 1, MPI::INT, partner_id, 0);
        // exchange upper chunks
        incoming_V.assign(incoming_size, 0);
        MPI::COMM_WORLD.Sendrecv(&upper.front(), upper.size(), MPI::INT, partner_id, 0,
                                 &incoming_V.front(), incoming_size, MPI::INT, partner_id, 0);
        // merge lower+incoming   
        V = lower;
        V.insert(V.end(), incoming_V.begin(), incoming_V.end());                     
                                 
    }
    else {
        // exchange sizes first
        int chunk_size = lower.size();
        MPI::COMM_WORLD.Sendrecv(&chunk_size, 1, MPI::INT, partner_id, 0,
                                 &incoming_size, 1, MPI::INT, partner_id, 0);
        // exchange lower chunks
        incoming_V.assign(incoming_size, 0);
        MPI::COMM_WORLD.Sendrecv(&lower.front(), lower.size(), MPI::INT, partner_id, 0,
                                 &incoming_V.front(), incoming_size, MPI::INT, partner_id, 0);    
        // merge incoming+upper        
        V = incoming_V;
        V.insert(V.end(), upper.begin(), upper.end());                     
    }
}


// MAIN
int main(int argc, char *argv[]) {
    
    vector<int> A,  chunkVect;
    int chunkSize;
    
    MPI::Init( argc, argv );

    int num_procs = MPI::COMM_WORLD.Get_size(); // Get number of procs.
    int id = MPI::COMM_WORLD.Get_rank(); // Get curr processor id/rank.

    if ( argc != 2 ) {
        cout <<" Usage: %s <input.txt>" << argv[0] << endl;
        return -1;
    }
    
    //ROOT: READ ARRAY AND SCATTER EVENLY AMONG PROCS
    if ( id == ROOT ) { 

        ifstream inFile( argv[1]) ;
        istream_iterator<int> start(inFile), end;
        A.assign(start, end);
        cout << "Read " << A.size() << " numbers" << endl;

#ifdef debug1
        cout << "numbers read in:\n";
        copy(A.begin(), A.end(), ostream_iterator<int>(cout, " "));
        cout << endl;
#endif    
        chunkSize = A.size()/num_procs;
    }
    
    // https://www.open-mpi.org/doc/v1.10/man3/MPI_Bcast.3.php
    MPI::COMM_WORLD.Bcast( &chunkSize, 1, MPI::INT, ROOT ); // only root can Bcast (id=0)
    chunkVect.assign(chunkSize, 0); // init vector size
#ifdef debug1
    cout << id << " : " << chunkSize << endl;
#endif    
   
    // https://www.open-mpi.org/doc/v1.10/man3/MPI_Scatter.3.php    
    MPI::COMM_WORLD.Scatter( &A[(id)*(chunkSize)], chunkSize, MPI::INT, // sending info -- only root can Scatter
                             &chunkVect.front(), chunkSize, MPI::INT,   // space for procs to receive
                             ROOT );    


#ifdef debug2
    cout << id << " : ";
    for(auto& x:chunkVect)
         cout << x << ' ';
    cout << endl;
#endif    
                                   
    HQsort(chunkVect, chunkSize);
   
#ifdef debug2
    cout << id << " : ";
    for(auto& x:chunkVect)
         cout << x << ' ';
    cout << endl;
#endif    
   
    MPI:: Finalize();
    return 0;
  
   
}

#include <algorithm> 
#include <cmath>       
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <mpi.h>

#define MASTER 0 
using namespace std;

int getMedian( const vector<int>& V ) { 
    const auto iter1 = V.begin() + V.size() / 2 - 1;
    const auto iter2 = V.begin() + V.size() / 2;
    auto median = (V.size() % 2 == 0) ? (*iter1 + *iter2) / 2 : *iter2;
    return median;
}

void HQsort (vector<int>& V, int& size) {
/* PARALLEL HYPERQUICKSORT (HQsort)    
    for i = d down to 1
        for each i-cube:
            Root of the i-cube broadcasts its median to all in the i-cube, to serve as pivot
            considering two (i-1)-subcubes of i-cube, each pair of partners in (i-1)-subcubes exchanges data:
                low-numbered PE gives its partner its data larger than pivot
                high-numbered PE gives its partner its data smaller than pivot
*/     
    int pivot, incoming_size;
    vector<int> incoming_V;

    int num_procs = MPI::COMM_WORLD.Get_size(); // Get number of procs.
    int curr_id = MPI::COMM_WORLD.Get_rank(); // Get curr processor id/rank.
 
    sort(V.begin(), V.end()); //locally sort proc vectors
    
    int dimensions = log(num_procs)/log(2);
    int level = num_procs;
    int partner_id;
    int root = num_procs/2;

    for (int d = dimensions ; d > 0; d--) { //TRANSVERSE I-CUBES
        MPI::COMM_WORLD.Barrier();
        
        level = level >> 1;
        if ( curr_id == root ) 
            pivot = getMedian(V);
       
        MPI::COMM_WORLD.Bcast( &pivot, 1, MPI::INT, root ); // Bcast pivot
        
        const auto iter = stable_partition( V.begin(), V.end(), [pivot](int n){ return n < pivot; } ); // ninja lambda       
        vector<int> lower(V.begin(), iter); // less than pivot 
        vector<int> upper(iter, V.end());  // equal to or more than pivot
        
        if( (curr_id & level) == 0 ) {  // LOWER SUBCUBE
            partner_id = curr_id + level;
            // exchange sizes first
            int chunk_size = upper.size();
            MPI::COMM_WORLD.Sendrecv(&chunk_size, 1, MPI::INT, partner_id, d,
                                     &incoming_size, 1, MPI::INT, partner_id, d);
            // send upper chunk
            incoming_V.assign(incoming_size, 0);
            MPI::COMM_WORLD.Sendrecv(&upper.front(), upper.size(), MPI::INT, partner_id, d,
                                     &incoming_V.front(), incoming_size, MPI::INT, partner_id, d);
            // merge lower + incoming   
            V.assign(lower.size()+incoming_V.size(), 0);
            merge(lower.begin(), lower.end(), incoming_V.begin(), incoming_V.end(), V.begin());                 
        }
        else {  // UPPER SUBCUBE
            partner_id = curr_id - level;
            // exchange sizes first
            int chunk_size = lower.size();
            MPI::COMM_WORLD.Sendrecv(&chunk_size, 1, MPI::INT, partner_id, d,
                                     &incoming_size, 1, MPI::INT, partner_id, d);
            // send lower chunk
            incoming_V.assign(incoming_size, 0);
            MPI::COMM_WORLD.Sendrecv(&lower.front(), lower.size(), MPI::INT, partner_id, d,
                                     &incoming_V.front(), incoming_size, MPI::INT, partner_id, d);    
            // merge incoming + upper        
            V.assign(incoming_V.size()+upper.size(), 0);
            merge(incoming_V.begin(), incoming_V.end(), upper.begin(), upper.end(), V.begin());                 
        }
        
        size = V.size();
        root = root >> 1;
    } //END FOR 
}

// MAIN
int main(int argc, char *argv[]) {
    
    vector<int> chunkVect;
    vector<int> A, displacement, count, D; // MASTER use only
    int chunkSize, local;
    
    MPI::Init( argc, argv );

    int num_procs = MPI::COMM_WORLD.Get_size(); // Get number of procs.
    int id = MPI::COMM_WORLD.Get_rank(); // Get curr processor id/rank.

    if ( argc != 2 ) {
        cout <<" Usage: %s <input.txt>" << argv[0] << endl;
        return -1;
    }
        
    //MASTER: READ ARRAY AND SCATTER EVENLY AMONG PROCS
    if ( id == MASTER ) { 
        ifstream inFile( argv[1]) ;
        istream_iterator<int> start(inFile), end;
        A.assign(start, end);
        cout << "Read " << A.size() << " numbers" << endl;
        chunkSize = A.size()/num_procs;

        D.assign(A.size(), 0);
        displacement.assign(num_procs, 0);
        count.assign(num_procs, 0);
    }
    
    MPI::COMM_WORLD.Bcast( &chunkSize, 1, MPI::INT, MASTER ); // Bcast chunk size
    chunkVect.assign(chunkSize, 0); // init vector size
   
    MPI::COMM_WORLD.Scatter( &A[(id)*(chunkSize)], chunkSize, MPI::INT, // split A per proc id
                             &chunkVect.front(), chunkSize, MPI::INT,   // space for procs to receive
                             MASTER );    
                                   
    HQsort(chunkVect, chunkSize);
    
    MPI::COMM_WORLD.Scan( &chunkSize, &local, 1, MPI::INT, MPI::SUM);
    
    MPI::COMM_WORLD.Gather(&local, 1, MPI::INT, &displacement[id+1], 1, MPI::INT, MASTER);
    MPI::COMM_WORLD.Gather(&chunkSize, 1, MPI::INT, &count[id], 1, MPI::INT, MASTER);
    
    MPI::COMM_WORLD.Gatherv(&chunkVect, chunkSize, MPI::INT, &D.front(), &count[id], &displacement[id], MPI::INT, MASTER);

    if ( id == MASTER ) { 
        for(auto& x:displacement) cout << x << '\t'; cout << endl;    
        for(auto& x:count) cout << x << '\t'; cout << endl;    

        ofstream outFile("./output.txt");
        for (const auto &e : D) outFile << e << " ";
    }
    
    MPI:: Finalize();
    return 0;   
}

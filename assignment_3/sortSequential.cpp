#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

//#define debug
using namespace std;
     
int pivot(vector<int>& a, int l, int u) {
    int v, i, j, tmp;
    v = a[l]; 
    i = l;   // lower bound iterator
    j = u+1; // upper bound iterator
    
    do {
        do // iterate until first unordered in left given v
            i++;
        while( a[i]<v && i<=u ); 
        
        do // iterate until  first unordered in right given v
            j--;
        while(v < a[j]); 
        
        if(i < j) {     // swap a[i] and a[j];
            tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        } 
        
    } while (i < j); // keep iterating until i and j cross
    
    a[l] = a[j]; 
    a[j] = v;
    
    return(j);
}

void quickSortSeq(vector<int>& a, int low, int upr) {
    if(low < upr) {
        int j = pivot(a, low, upr);
        quickSortSeq(a, low, j-1); 
        quickSortSeq(a, j+1, upr);
    }
}

// MAIN
int main(int argc, char *argv[]) {
    
    if ( argc != 2 ) {
        cout <<" Usage: %s <input.txt>" << argv[0] << endl;
        return -1;
    }

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
    
    quickSortSeq(A, 0, A.size()-1);
   
    ofstream outFile("./output_serial.txt");
    for (const auto& e : A) outFile << e << " ";
    
    return 0;
  
   
}

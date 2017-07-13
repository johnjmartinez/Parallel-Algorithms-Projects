#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <vector>

//#define debug
using namespace std;
     
int chunk(vector<int> &a, int l, int u) {
    int v, i, j, temp;
    v = a[l];
    i = l;
    j = u+1;
    
    do {
        do
            i++;
        while( a[i]<v && i<=u);
        
        do
            j--;
        while(v  <a[j]);
        
        if(i < j) {
            temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
        
    } while(i < j);
    
    a[l] = a[j];
    a[j] = v;
    
    return(j);
}

void quickSortSeq(vector<int> &a, int low, int upr) {
    if(low < upr) {
        int j = chunk(a, low, upr);
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
   
    for(auto x:A)
         cout << ' ' << x;
    cout << endl;
    return 0;
  
   
}

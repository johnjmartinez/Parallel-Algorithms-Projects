- To compile run:
    ./compileIt.sh <file.cu> --> <file_> (executable)

- GPU SORTING ALGORITHMS

 * radixSort.cu          - basic radix sort implementation
 * brickSort.cu          - basic brick sort implementation
 * brickSort2.cu         - shared mem brick sort+merge implementation
 * brickSort3.cu         - cub blk sort + brick merge implementation 

 * thrustSortSimple.cu   - simple gpu sort using thrust library
 * thrustMergeSort.cu    - mergesort using thrust library
 
 * cubBlkRadixSort.cu    - block radix + dev merge sort using cub library
 * cubDevRadixSort.cu    - simple radix sort using cub library 
 * cubDevRadixSort2.cu   - byte based device radix sort using cub library
 
 
 - Creating logs by running, i.e.  (shell script below)
    function run_sorts() {  
        for x in *_; do echo -e "\n==== $x"; ./$x $val; done
    }
    
    tag=<gpu> ... ie GT720
    val=$((1<<16)) #64K
    run_sorts &> misc/${tag}.log.${val}

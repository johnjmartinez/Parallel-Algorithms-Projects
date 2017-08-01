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
 
 
 - Creating logs by running : (shell script below)
    function run_sorts() {  
    
        for x in `ls *_ | grep -v cubBlk` ; do 
            echo -e "\n==== $x"; ./$x $val
        done
        
        x=cubBlkRadixSort_
        for n in `seq 0 5`; do 
            echo -e "\n==== $x $n"; ./$x $val $n
        done
    }
    
    tag=GTX1050Ti #<GPU>
    for i in `seq 19 25`; do
        val=$((1<<${i})) 
        run_sorts &> misc/${tag}.log.${val}
    done

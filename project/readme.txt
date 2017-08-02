- To compile run:
    ./compileIt.sh <file.cu> --> <file_> (executable)

- GPU SORTING ALGORITHMS

 * radixSort.cu          - basic radix sort implementation (non-optimized)
 * brickSort.cu          - basic brick sort implementation (non-optimized)
 * brickSort2.cu         - shared mem brick sort+merge implementation
 * brickSort3.cu         - cub blk sort + brick merge implementation 

 * thrustSortSimple.cu   - simple gpu sort using thrust library
 * thrustMergeSort.cu    - mergesort using thrust library
 
 * cubBlkRadixSort.cu    - block radix + dev merge sort using cub library
 * cubDevRadixSort.cu    - simple radix sort using cub library 
 * cubDevRadixSort2.cu   - byte based device radix sort using cub library
 
 
 - Creating logs by running : (batch job below)
    function run_sorts() {

        for x in `ls *_ | egrep -v "cubBlk|quickS"` ; do
            echo -e "\n==== $x"; ./$x $val
        done

        x=cubBlkRadixSort_
        for n in `seq 0 5`; do
            echo -e "\n==== $x $n"; ./$x $val $n
        done
    }

    tag=TeslaK20m #<GPU>
    for i in `seq 15 25`; do
        val=$((1<<${i}))
        date
        echo -e "\n Running seq $i val $val ... "
        run_sorts &> misc/${tag}.log.${val}
    done

    x=quickSort2_   # had to separate it due to odd behavior in stampede
    for i in `seq 15 25`; do
        val=$((1<<${i}))
        date
        echo -e "\n Running quickSort2_ seq $i val $val ... "
        echo -e "\n==== $x" >> misc/${tag}.log.${val}
        ./$x $val >> misc/${tag}.log.${val}
    done

- To compile run:
    ./compileIt.sh <file.cu> --> <file_>

- SORTING ALGORITHMS

 * radixSort.cu          - basic radix gpu sort implementation
 * brickSort.cu          - basic brick gpu sort implementation
 
 * thrustSortSimple.cu   - radix gpu sort using thrust library sample 1
 * thrustMergeSort.cu    - merge gpu sort using thrust library sample 2
 * cubDevRadixSort.cu    - device radix sort using cub library
 * cubBlkRadixSort.cu    - block radix + dev merge sort using cub library

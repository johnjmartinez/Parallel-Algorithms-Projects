

algs="quickSort2 brickSort2 brickSort3 thrustSortSimple thrustOptmSort cubDevRadixSort_ mgpuMergeSort"  

rm $g
touch $g
for i in `seq 15 25`; do 
    n=$((1<<i))
    echo -e "$g.log.$n\t"> tmp
    for a in $algs; do 
        t=$(grep -A4 $a $g.log.$n | grep Thro | perl -pe "s/Throughput =\s+//; s/Elements//; s/,\s+Time.+//") 
        echo -e "$a\t$t\t"  >> tmp
    done

    paste $g tmp > x
    mv x $g
done 


cubDevRadixSort2_ 
cubBlkRadixSort_ 

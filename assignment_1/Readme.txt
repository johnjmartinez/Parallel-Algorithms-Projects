Q1(a) both seq and par - array_minimum_par.cu
Q1(b) both seq and par - array_output.cu
Q2(a,b,c) - seq(rangeCountSequential.cpp), par(rangeCount.cu)
Q3 both seq and par - radix_sort.cu
range.bash - bash file for compiling and running Q2.
run_batch - modified batch file for job submission to stampede

Please change the nthreads value to match the number of elements in the array.I modified the batch file(run_batch) to accept 2 command line arg( after output put $1 $2), so the batch file needs to be run as eg – sbatch run_batch file_450.txt 450  ( 450 is the number records in the file and file_450.txt is the file name)
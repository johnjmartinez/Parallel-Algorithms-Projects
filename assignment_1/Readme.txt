array_minimum_par.cu  		- Q1(a) both sequential and parallel
array_output.cu   		- Q1(b) both sequential and parallel
rangeCountSequential.cpp  	- Q2(a,b,c) sequential.
rangeCount.cu   		- Q2(a,b,c) parallel
radix_sort.cu   		- Q3 both sequential and parallel
file_450.txt  			- input file with 450 records 
range.bash 			- bash file for compiling and running Q2
run_batch 			- modified batch file for job submission to stampede for Q1(a and b) and Q3

For Q1(a and b) and Q3 - Please change the nthreads value to match the number of elements in the array.I modified the batch file(run_batch) to accept 2 command line arg( after output put $1 $2), so the batch file needs to be run as eg – sbatch run_batch file_450.txt 450  ( file_450.txt is the file name and 450 is the number records in the file)
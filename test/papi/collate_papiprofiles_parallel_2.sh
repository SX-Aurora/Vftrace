#!/bin/bash

nmpi=4 # Nr. of MPI ranks

function check_ranksum () {
  funcname=$1
  outfile=$2
  n1=`grep $funcname $outfile | head -1 | awk '{print $6}'`
  n2=`grep $funcname $outfile | tail -1 | awk '{print $6}'`
  nn=`bc <<< "$n1 * $nmpi"`
  
  if [ "$nn" -ne "$nmpi" ]; then
     echo "rank sum for $funcname does not match: $nn, expected $n2"
     exit 1
  fi
}

set -x
test_name=collate_papiprofiles_parallel_2
exe_name=collate_papiprofiles_parallel_2
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${output_file}

rm -f ${output_file}
${MPI_EXEC} ${MPI_OPTS} ${NP} ${nmpi} ./${exe_name} > ${output_file} || exit 1

check_ranksum "func0" $output_file
check_ranksum "papifunc1" $output_file
check_ranksum "papifunc2" $output_file

exit 0

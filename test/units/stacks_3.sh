#!/bin/bash
set -x
test_name=stacks_3
output_file=$test_name.out
nprocs=4

if [ "x$HAS_MPI" == "xYES" ]; then
   ref_file=${srcdir}/ref_output/mpi/$test_name.out
fi

if [ "x$HAS_MPI" == "xYES" ]; then
   > ${output_file}
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${test_name} || exit 1

   for irank in $(seq 0 1 $(bc <<< "${nprocs}-1"));
   do
      tmpoutfile=$(echo "stacks_2_rank${irank}_tmp.out")
      if [ -e ${tmpoutfile} ]; then
         cat ${tmpoutfile} >> ${output_file}
         rm -f ${tmpoutfile}
      else
         echo "Output file of rank ${irank} missing! Aborting!"
         exit 1
      fi
   done
fi

diff $ref_file $output_file

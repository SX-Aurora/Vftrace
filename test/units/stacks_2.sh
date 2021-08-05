#!/bin/bash
set -x
test_name=stacks_2
output_file=$test_name.out
nprocs=4

if [ "x$HAS_MPI" == "xYES" ]; then
   ref_file=${srcdir}/ref_output/mpi/$test_name.out
fi

if [ "x$HAS_MPI" == "xYES" ]; then
   > ${output_file}
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${test_name}
   last_success=$?

   if [ $last_success == 0 ]; then
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
fi

if [ $last_success == 0 ]; then
  # There is one temporary output file for each rank.
  # We just put one after the other.
  # cat ${output_file}_0 ${output_file}_1 ${output_file}_2 ${output_file}_3  > $output_file
  diff $ref_file $output_file
else
  exit  $last_success
fi



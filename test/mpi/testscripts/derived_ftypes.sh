#!/bin/bash

source ${srcdir}/../../environment/filenames.sh

vftr_binary=derived_ftypes
configfile=${vftr_binary}.json
nprocs=2

determine_bin_prefix $vftr_binary

echo "{\"sampling\": {\"active\": true}}" > ${configfile}
export VFTR_CONFIG=${configfile}

${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1

mpitype=MPI_DERIVED_TYPE

for ivfd in $(seq 0 1 $(bc <<< "${nprocs}-1"));
do
   vfdfile=$(get_vfdfile_name ${vftr_binary} ${ivfd})
   ../../../tools/vftrace_vfd_dump ${vfdfile}

   tmptype=$(../../../tools/vftrace_vfd_dump ${vfdfile} | \
             awk '($2=="send" || $2=="recv") && $3!="end"{getline;print;}' | \
             sed 's/=/ /g;s/(/ /g' | \
             awk '{print $4}' | \
             head -n 1)

   if [ ! "${mpitype}" = "${tmptype}" ] ; then
      echo "Expected MPI_TYPE ${mpitype} but ${tmptype} was used."
      exit 1;
   fi
done

#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
vftr_binary=hwprof_region4
configfile=${vftr_binary}.json
nprocs=1

determine_bin_prefix $vftr_binary

logfile=$(get_logfile_name ${vftr_binary} "all")
vfdfile=$(get_vfdfile_name ${vftr_binary} "0")

configfile=${test_name}.json
cat << EOF > ${configfile}
{
   "mpi": {"show_table": false},
   "sampling": {
      "active": true,
      "precise_functions": "do_region|main"
   },
   "hwprof": {
      "type": "dummy",
      "counters": [
          {
             "hwc_name": "test",
             "symbol": "t"
          }
      ]
   }
}
EOF
export VFTR_CONFIG=${configfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

counter=0
for c in `../../tools/vftrace_vfd_dump ${vfdfile} | awk '$3=="exit" || $3=="call" {print $2}'`; do
  if [ "$c" != "$counter" ]; then
     echo "Registered HW counter does not match!"
     echo "Expected $counter, read $c"
     exit 1
  fi
  counter=$(($((counter+1))%10))
done

### The same, but with inactive hardware profiling. We expect no entries in the vfd dump.
cat << EOF > ${configfile}
{
   "mpi": {"show_table": false},
   "sampling": {
      "active": true,
      "precise_functions": "do_region|main"
   },
   "hwprof": {
      "type": "dummy",
      "active": false,
      "counters": [
          {
             "hwc_name": "test",
             "symbol": "t"
          }
      ]
   }
}
EOF

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

# Without hardware counters, only three columns exist in the sample: Timestamp, kind & stack
nfound=`../../tools/vftrace_vfd_dump ${vfdfile} | awk '$2=="exit" || $2=="call" {print NF}' | head -1`
if [ "$nfound" != "3" ]; then
   echo "vfd columns without HW profiling do not match!"
   echo "Expected 3, found $nfound"
   exit -1
fi

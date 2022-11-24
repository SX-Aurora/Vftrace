#!/bin/bash
set -x

source ${srcdir}/../environment/filenames.sh

test_name=acc_region4
vftr_binary=${test_name}
configfile=${test_name}.json

logfile=$(get_logfile_name ${vftr_binary} "all")

cat << EOF > ${configfile}
{
   "openacc": {
        "show_event_details": true
    }
}
EOF
export VFTR_CONFIG=${configfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
fi

## Search for wait event
region_id=0xd457bcaaa54b77d2
grep ${region_id} ${logfile} | grep wait
if [ "$?" -ne 0 ]; then
   echo "Wait event not found."
   exit 1
fi

## Wait event is called once
ncalls=`grep ${region_id} ${logfile} | grep wait | awk '{print $8}'`
if [ "${ncalls}" -ne 1 ]; then
   echo "'wait' is not called exactly once."
   exit 1;
fi

## Search for wait event
region_id=0x49fe93be4721c93a
grep ${region_id} ${logfile} | grep wait
if [ "$?" -ne 0 ]; then
   echo "Wait event not found."
   exit 1
fi

## Wait event is called once
ncalls=`grep ${region_id} ${logfile} | grep wait | awk '{print $8}'`
if [ "${ncalls}" -ne 1 ]; then
   echo "'wait' is not called exactly once."
   exit 1;
fi

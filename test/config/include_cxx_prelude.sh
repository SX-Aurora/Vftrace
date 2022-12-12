#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x

test_name=include_cxx_prelude
configfile=${test_name}.json
determine_bin_prefix $test_name
logfile=$(get_logfile_name ${test_name} "all")

rm -f ${logfile}
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} || exit 1
else
   ./${test_name} || exit 1
fi


cat ${logfile}
# Count how often the before_main function appears in the profile
count=$(cat ${logfile} | grep "init_nelems_before_main" | wc -l)
if [[ "${count}" -ne "0" ]] ; then
   echo "Expected not to find \"init_nelems_before_main\" in logfile, but found it ${count} times."
   exit 1;
fi

echo "{\"include_cxx_prelude\": true}" > ${configfile}
export VFTR_CONFIG=${configfile}
rm -f ${logfile}
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} || exit 1
else
   ./${test_name} || exit 1
fi
cat ${logfile}
# Count how often the before_main function appears in the profile
count=$(cat ${logfile} | grep "init_nelems_before_main" | wc -l)
if [[ "${count}" -lt "2" ]] ; then
   echo "Expected to find \"init_nelems_before_main\" at least two times in logfile, but found it ${count} times."
   exit 1;
fi


#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=demangle_cxx
configfile=${test_name}.json
logfile=$(get_logfile_name ${test_name} "all")

rm -f ${logfile}
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} || exit 1
else
   ./${test_name} || exit 1
fi
for functions in quicksort issorted;
do
   nfunction=$(grep -i " ${functions} " ${logfile} | wc -l)
   if [ "${nfunction}" -ne "0" ] ; then
      echo "Expected the demangled function \"${function}\""
      echo "not to appear in logfile \"${logfile}\","
      echo "but found it ${nfunction} times"
      exit 1
   fi
done

echo "{\"demangle_cxx\": true}" > ${configfile}
export VFTR_CONFIG=${configfile}
rm -f ${logfile}
if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} || exit 1
else
   ./${test_name} || exit 1
fi
for functions in quicksort issorted;
do
   nfunction=$(grep -i " ${functions} " ${logfile} | wc -l)
   if [ "${nfunction}" -ne "2" ] ; then
      echo "Expected the demangled function \"${function}\""
      echo "to appear twice in logfile \"${logfile}\","
      echo "but found it ${nfunction} times"
      exit 1
   fi
done

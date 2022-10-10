#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
vftr_binary=vftr_demangle_cxx
logfile=$(get_logfile_name ${vftr_binary} "all")

rm -f ${logfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
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

export VFTR_DEMANGLE_CXX="yes"
rm -f ${logfile}

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${vftr_binary} || exit 1
else
   ./${vftr_binary} || exit 1
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

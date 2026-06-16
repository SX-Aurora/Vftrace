#!/bin/bash

source ${srcdir}/../environment/filenames.sh

vftr_binary=fpause_resume
configfile=${vftr_binary}.json
nprocs=1

determine_bin_prefix $vftr_binary

echo "{\"sampling\": {\"active\": true, \"precise_functions\": \"fkt*\"}, \"papi\": {\"show_tables\": false}}" > ${configfile}
export VFTR_CONFIG=${configfile}

logfile=$(get_logfile_name ${vftr_binary} "all")
vfdfile=$(get_vfdfile_name ${vftr_binary} "0")

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${maxnreg} || exit 1
else
   ./${vftr_binary} ${maxnreg} || exit 1
fi

cat ${logfile}

# With the Intel LLVM compiler, there are additional symobls ending with `_.void', one for 
# each actual symbol. Therefore, we expect twice as much occurences of fkt1 and fkt3 than with
# other compilers.
if [ "x${USES_INTEL_COMPILER}" == "xYES" ]; then
   n_expect=4
else
   n_expect=2
fi
# Check for nr. of occurences of fkt1.
inprof=$(cat ${logfile} | \
         grep " fkt1" | \
         wc -l)
if [ "${inprof}" -ne "${n_expect}" ] ; then
   echo "Mismatch in nr. of \"fkt1\" entries."
   echo "Expected ${n_expect}, found ${inprof}"
   exit 1;
fi
# Check for nr. of occurences of fkt2.
inprof=$(cat ${logfile} | \
         grep " fkt2" | \
         wc -l)
if [ "${inprof}" -gt "0" ] ; then
   echo "Mismatch in nr. of \"fkt2\" entries."
   echo "Expected 0, found ${inprof}"
   exit 1;
fi
# Check for nr. of occurences of fkt3.
inprof=$(cat ${logfile} | \
         grep " fkt3" | \
         wc -l)
if [ "${inprof}" -ne "${n_expect}" ] ; then
   echo "Mismatch in nr. of \"fkt3\" entries."
   echo "Expected ${n_expect}, found ${inprof}"
   exit 1;
fi
# check for existance of vftrace_pause
inprof=$(cat ${logfile} | \
         grep " vftrace_pause" | \
         wc -l)
if [ "${inprof}" -ne "2" ] ; then
   echo "Mismatch in nr. of \"vftrace_pause\" entries."
   echo "Expected 2, found ${inprof}"
   exit 1;
fi


../../tools/vftrace_vfd_dump ${vfdfile}

# check for occurences in the vfd file
if [ "x${USES_INTEL_COMPILER}" == "xYES" ]; then
   n_expect=2
else
   n_expect=1
fi
ncalls=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "call fkt1" | \
         wc -l)
if [ "${ncalls}" -ne "${n_expect}" ] ; then
   echo "Mismatch in nr. of \"fkt1-entry\" entries in the vfd file."
   echo "Expected ${n_expect}, found ${inprof}"
   exit 1;
fi
nexits=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "exit fkt1" | \
         wc -l)
if [ "${nexits}" -ne "${n_expect}" ] ; then
   echo "Mismatch in nr. of \"fkt1-exit\" entries in the vfd file."
   echo "Expected ${n_expect}, found ${inprof}"
   exit 1;
fi

ncalls=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "call fkt2" | \
         wc -l)
if [ "${ncalls}" -gt "0" ] ; then
   echo "Mismatch in nr. of \"fkt2-entry\" entries in the vfd file."
   echo "Expected 0, found ${inprof}"
   exit 1;
fi
nexits=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "exit fkt2" | \
         wc -l)
if [ "${nexits}" -gt "0" ] ; then
   echo "Mismatch in nr. of \"fkt2-exit\" entries in the vfd file."
   echo "Expected 0, found ${inprof}"
   exit 1;
fi

ncalls=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "call fkt3" | \
         wc -l)
if [ "${ncalls}" -ne "${n_expect}" ] ; then
   echo "Mismatch in nr. of \"fkt3-entry\" entries in the vfd file."
   echo "Expected ${n_expect}, found ${inprof}"
   exit 1;
fi
nexits=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "exit fkt3" | \
         wc -l)
if [ "${nexits}" -ne "${n_expect}" ] ; then
   echo "Mismatch in nr. of \"fkt3-exit\" entries in the vfd file."
   echo "Expected ${n_expect}, found ${inprof}"
   exit 1;
fi

ncalls=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "call vftrace_pause" | \
         wc -l)
if [ "${ncalls}" -ne "1" ] ; then
   echo "Mismatch in nr. of \"vftrace_pause\" entries in the vfd file."
   echo "Expected 1, found ${inprof}"
   exit 1;
fi

#!/bin/bash

source ${srcdir}/../environment/filenames.sh

vftr_binary=fpause_resume
nprocs=1

logfile=$(get_logfile_name ${vftr_binary} "all")
vfdfile=$(get_vfdfile_name ${vftr_binary} "0")

export VFTR_SAMPLING="Yes"
export VFTR_PRECISE="fkt*"

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} ${maxnreg} || exit 1
else
   ./${vftr_binary} ${maxnreg} || exit 1
fi

cat ${logfile}

# check for existance of fkt1
inprof=$(cat ${logfile} | \
         grep " fkt1" | \
         wc -l)
if [ "${inprof}" -ne "2" ] ; then
   echo "Function \"fkt1\" not found in log file the expected amount"
   exit 1;
fi
# check for existance of fkt2
inprof=$(cat ${logfile} | \
         grep " fkt2" | \
         wc -l)
if [ "${inprof}" -gt "0" ] ; then
   echo "Function \"fkt2\" should not appear in log file"
   exit 1;
fi
# check for existance of fkt3
inprof=$(cat ${logfile} | \
         grep " fkt3" | \
         wc -l)
if [ "${inprof}" -ne "2" ] ; then
   echo "Function \"fkt3\" not found in log file the expected amount"
   exit 1;
fi
# check for existance of vftrace_pause
inprof=$(cat ${logfile} | \
         grep " vftrace_pause" | \
         wc -l)
if [ "${inprof}" -ne "2" ] ; then
   echo "Function \"vftrace_pause\" not found in log file the expected amount"
   exit 1;
fi


../../tools/vftrace_vfd_dump ${vfdfile}

# check for existance of fkt1
ncalls=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "call fkt1" | \
         wc -l)
if [ "${ncalls}" -ne "1" ] ; then
   echo "Call to function \"fkt1\" not found in vfd file"
   exit 1;
fi
nexits=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "exit fkt1" | \
         wc -l)
if [ "${nexits}" -ne "1" ] ; then
   echo "Exit from function \"fkt1\" not found in vfd file"
   exit 1;
fi
# check for existance of fkt2
ncalls=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "call fkt2" | \
         wc -l)
if [ "${ncalls}" -gt "0" ] ; then
   echo "Call to function \"fkt2\" should not appear in vfd file"
   exit 1;
fi
nexits=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "exit fkt2" | \
         wc -l)
if [ "${nexits}" -gt "0" ] ; then
   echo "Exit from function \"fkt2\" should not appear in vfd file"
   exit 1;
fi
# check for existance of fkt3
ncalls=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "call fkt3" | \
         wc -l)
if [ "${ncalls}" -ne "1" ] ; then
   echo "Call to function \"fkt3\" not found in vfd file"
   exit 1;
fi
nexits=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "exit fkt3" | \
         wc -l)
if [ "${nexits}" -ne "1" ] ; then
   echo "Exit from function \"fkt3\" not found in vfd file"
   exit 1;
fi
# check for existance of vftrace_pause
ncalls=$(../../tools/vftrace_vfd_dump ${vfdfile} | \
         grep "call vftrace_pause" | \
         wc -l)
if [ "${ncalls}" -ne "1" ] ; then
   echo "Call to function \"vftrace_pause\" not found in vfd file"
   exit 1;
fi

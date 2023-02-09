#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x

test_name=signal_handling
output_file=${test_name}.out
logfile=${test_name}_all.log

determine_bin_prefix ${test_name}
logfile=${BIN_PREFIX}${logfile}

function run_binary () {
   testcase=$1
   if [ "x${HAS_MPI}" == "xYES" ]; then
      ${MPI_EXEC} ${MPI_OPTS} ${NP} 1 ./${test_name} ${testcase} > ${output_file}
   else
      ./${test_name} ${testcase} > ${output_file}
   fi
}

#0: Terminated
#1: Interrupt
#2: Aborted
#3: Floating point exception
#4: Quit
#5: Segmentation fault

run_binary 0
grep "signal: Terminated" $logfile
if [ "$?" -ne "0" ]; then
   echo "Expected SIGTERM"
   exit 1
fi

run_binary 1
grep "signal: Interrupt" $logfile
if [ "$?" -ne "0" ]; then
   echo "Expected SIGINT"
   exit 1
fi

run_binary 2
grep "signal: Aborted" $logfile
if [ "$?" -ne "0" ]; then
   echo "Expected SIGABRT"
   exit 1
fi

run_binary 3
grep "signal: Floating point exception" $logfile
if [ "$?" -ne "0" ]; then
   echo "Expected SIGFPE"
   exit 1
fi

run_binary 4
grep "signal: Quit" $logfile
if [ "$?" -ne "0" ]; then
   echo "Expected SIGQUIT"
   exit 1
fi

run_binary 5
grep "signal: Segmentation fault" $logfile
if [ "$?" -ne "0" ]; then
   echo "Expected SIGSEGV"
   exit 1
fi

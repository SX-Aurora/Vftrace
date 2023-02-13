#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=separate_ranklogfiles
configfile=${test_name}.json
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

determine_bin_prefix $test_name
logfile0=$(get_logfile_name $test_name "0")
logfile1=$(get_logfile_name $test_name "1")

# create logfile -> only one logfile
cat << EOF > ${configfile}
{
   "logfile_for_ranks": "all",
   "profile_table": {
      "separate": false,
      "show_minmax_summary": true
   },
   "name_grouped_profile_table": {
      "show_table": true
   },
   "mpi": {
      "show_table": true
   }
}
EOF
export VFTR_CONFIG=${configfile}

rm -f ${output_file}

${MPI_EXEC} ${MPI_OPTS} ${NP} 2 ./${test_name} > ${output_file} || exit 1

n=`grep "Runtime\ profile" ${logfile0} | wc -l`
if [ "$n" != "2" ]; then
  echo "Error: Expected string 'Runtime profile' 2 times in $logfile0, but found it $n times"
  exit 1
fi
n=`grep "Runtime\ profile" ${logfile1} | wc -l`
if [ "$n" != "2" ]; then
  echo "Error: Expected string 'Runtime profile' 2 times in $logfile1, but found it $n times"
  exit 1
fi


nmpi=`grep 'Communication\ profile' ${logfile0} | wc -l`
if [ "$nmpi" != 1 ]; then
  echo "Error: Expected string 'Communication profile' once in $logfile0, but found it $n times"
  exit 1
fi
nmpi=`grep 'Communication\ profile' ${logfile1} | wc -l`
if [ "$nmpi" != 1 ]; then
  echo "Error: Expected string 'Communication profile' once in $logfile1, but found it $n times"
  exit 1
fi


# create logfile -> separate logfiles
cat << EOF > ${configfile}
{
   "logfile_for_ranks": "all",
   "profile_table": {
      "separate": true,
      "show_minmax_summary": true
   },
   "name_grouped_profile_table": {
      "show_table": true
   },
   "mpi": {
      "show_table": true
   }
}
EOF
export VFTR_CONFIG=${configfile}

rm -f ${output_file}

${MPI_EXEC} ${MPI_OPTS} ${NP} 2 ./${test_name} > ${output_file} || exit 1

mpi_logfile0=$(get_logfile_name $test_name "0_mpi")
mpi_logfile1=$(get_logfile_name $test_name "1_mpi")

n=`grep "Runtime\ profile" ${logfile0} | wc -l`
if [ "$n" != "2" ]; then
  echo "Error: Expected string 'Runtime profile' 2 times in $logfile0, but found it $n times"
  exit 1
fi
n=`grep "Runtime\ profile" ${logfile1} | wc -l`
if [ "$n" != "2" ]; then
  echo "Error: Expected string 'Runtime profile' 2 times in $logfile1, but found it $n times"
  exit 1
fi


nmpi=`grep 'Communication\ profile' ${mpi_logfile0} | wc -l`
if [ "$nmpi" != 1 ]; then
  echo "Error: Expected string 'Communication profile' once in $mpi_logfile0, but found it $n times"
  exit 1
fi
nmpi=`grep 'Communication\ profile' ${mpi_logfile1} | wc -l`
if [ "$nmpi" != 1 ]; then
  echo "Error: Expected string 'Communication profile' once in $mpi_logfile1, but found it $n times"
  exit 1
fi

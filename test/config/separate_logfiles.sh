#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
test_name=separate_logfiles
configfile=${test_name}.json
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

determine_bin_prefix $test_name
all_logfile=$(get_logfile_name $test_name "all")
group_logfile=$(get_logfile_name $test_name "all_namegroup")
minmax_logfile=$(get_logfile_name $test_name "all_minmax")
mpi_logfile=$(get_logfile_name $test_name "all_mpi")

# create logfile -> only one logfile
cat << EOF > ${configfile}
{
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

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 2 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

n=`grep "Runtime\ profile" ${all_logfile} | wc -l`
if [ "$n" != "3" ]; then
  echo "Error: Expected string 'Runtime profile' 3 times in $all_logfile, but found it $n times"
  exit 1
fi

n=`grep "Min/Max" ${all_logfile} | wc -l`
if [ "$n" != "1" ]; then
    echo "Error: Expected string 'Min/Max' once in $minmax_logfile, but found it $n times"
    exit 1
fi

if [ "x${HAS_MPI}" == "xYES" ]; then
  nmpi=`grep 'Communication\ profile' ${all_logfile} | wc -l`
  if [ "$nmpi" != 1 ]; then
    echo "Error: Expected string 'Communication profile' once in $mpi_logfile, but found it $n times"
    exit 1
  fi
fi

# create logfile -> separate logfiles
cat << EOF > ${configfile}
{
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

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} 2 ./${test_name} > ${output_file} || exit 1
else
   ./${test_name} > ${output_file} || exit 1
fi

check_file_exists $all_logfile
n=`grep "Runtime\ profile" ${all_logfile} | wc -l`
if [ "$n" != "2" ]; then
  echo "Error: Expected string 'Runtime profile' 2 times in $all_logfile, but found it $n times"
  exit 1
fi
check_file_exists $group_logfile
n=`grep "Runtime\ profile" ${all_logfile} | wc -l`
if [ "$n" != "2" ]; then
  echo "Error: Expected string 'Runtime profile' 2 times in $group_logfile, but found it $n times"
  exit 1
fi
check_file_exists $minmax_logfile
n=`grep "Min/Max" ${minmax_logfile} | wc -l`
if [ "$n" != "1" ]; then
    echo "Error: Expected string 'Min/Max' once in $minmax_logfile, but found it $n times"
    exit 1
fi

if [ "x${HAS_MPI}" == "xYES" ]; then
  check_file_exists $mpi_logfile
  nmpi=`grep 'Communication\ profile' ${mpi_logfile} | wc -l`
  if [ "$nmpi" != 1 ]; then
    echo "Error: Expected string 'Communication profile' once in $mpi_logfile, but found it $n times"
    exit 1
  fi
fi

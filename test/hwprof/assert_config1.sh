#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x
vftr_binary=assert_config1
configfile=${vftr_binary}.json
errfile=${vftr_binary}.err
nprocs=1

determine_bin_prefix $vftr_binary

# This test checks that builtin symbols are correctly caught by the assertion.
# This is done by searching for the mesage "is reserved for a builtin variable"
# in the error output.
# The first test uses a valid symbol, so the string must not appear.
# The next two tests try to register pre-defined symbols, and must veto that.

export VFTR_CONFIG=${configfile}

cat << EOF > ${configfile}
{
   "mpi": {"show_table": false},
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

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} 2> ${errfile}
else
   ./${vftr_binary} 2> ${errfile}
fi

if [ "$(grep is\ reserved\ for\ a\ builtin\ variable $errfile | wc -l)" != "0" ]; then
   exit 1
fi


cat << EOF > ${configfile}
{
   "mpi": {"show_table": false},
   "hwprof": {
      "type": "dummy",
      "counters": [
          {
             "hwc_name": "test",
             "symbol": "T"
          }
      ]
   }
}
EOF

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary} 2> ${errfile}
else
   ./${vftr_binary} 2> ${errfile}
fi

if [ "$(grep is\ reserved\ for\ a\ builtin\ variable $errfile | wc -l)" != "1" ]; then
   exit 1
fi

cat << EOF > ${configfile}
{
   "mpi": {"show_table": false},
   "hwprof": {
      "type": "dummy",
      "counters": [
          {
             "hwc_name": "test",
             "symbol": "NCALLS"
          }
      ]
   }
}
EOF

if [ "x${HAS_MPI}" == "xYES" ]; then
   ${MPI_EXEC} ${MPI_OPTS} ${NP} ${nprocs} ./${vftr_binary}
else
   ./${vftr_binary}
fi

if [ "$(grep is\ reserved\ for\ a\ builtin\ variable $errfile | wc -l)" != "1" ]; then
   exit 1
fi


#!/bin/bash

function determine_bin_prefix () {
   exefile=$1
   grep "program=lt-" $exefile
   if [ "$?" -eq "0" ]; then
      export BIN_PREFIX="lt-"
   else
      export BIN_PREFIX=""
   fi
}

function get_logfile_name () {
   logfile="$1_$2".log
   echo ${BIN_PREFIX}${logfile}
}

function get_vfdfile_name () {
   vfdfile="$1_$2".vfd
   echo ${BIN_PREFIX}${vfdfile}
}

function rm_outfiles() {
   output_file=$1
   error_file=$2
   test_name=$3
   rm -f ${output_file}
   rm -f ${error_file}
   for file in ${test_name}_*.log ${test_name}_*.vfd; do
      ##if [ "${IS_SHARED_BUILD}" == "YES" ]; then
      ##  file=lt-$file
      ##fi
      rm -f ${BIN_PREFIX}${file}
   done
}

function check_file_exists() {
   filename=$1
   if [ ! -f ${filename} ] ; then
      echo "Could not find file: \"${filename}\"!"
      exit 1
   fi
}

function check_file_notexists() {
   filename=$1
   if [ -f ${filename} ] ; then
      echo "File \"${filename}\" does exist although it should not!"
      exit 1
   fi
}

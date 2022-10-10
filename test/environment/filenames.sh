#!/bin/bash

function get_logfile_name () {
   logfile="$1_$2".log
   if [ "${IS_SHARED_BUILD}" == "YES" ]; then
      logfile=lt-$logfile
   fi
   echo $logfile
}

function get_vfdfile_name () {
   vfdfile="$1_$2".vfd
   if [ "${IS_SHARED_BUILD}" == "YES" ]; then
      vfdfile=lt-$vfdfile
   fi
   echo $vfdfile
}

function rm_outfiles() {
   output_file=$1
   error_file=$2
   test_name=$3
   rm -f ${output_file}
   rm -f ${error_file}
   for file in ${test_name}_*.log ${test_name}_*.vfd; do
      if [ "${IS_SHARED_BUILD}" == "YES" ]; then
        file=lt-$file
      fi
      rm -f $file
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

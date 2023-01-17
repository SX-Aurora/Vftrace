#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x

test_name=calculator
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${output_file}

./calculator > ${output_file}

diff ${output_file} ${ref_file} || exit 1

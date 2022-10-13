#!/bin/bash
set -x
test_name=accumulate_cuptiprof
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${test_name}.out

rm -f ${output_file}

./accumulate_cuptiprof > ${output_file} || exit 1

diff ${output_file} ${ref_file}

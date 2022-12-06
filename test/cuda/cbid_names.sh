#!/bin/bash
set -x
test_name=cbid_names
output_file=${test_name}.out
ref_file=${srcdir}/ref_output/${output_file}

rm -f ${output_file}

./cbid_names > ${output_file} || exit 1

diff ${output_file} ${ref_file}

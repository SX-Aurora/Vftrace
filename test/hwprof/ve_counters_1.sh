#!/bin/bash

source ${srcdir}/../environment/filenames.sh

set -x

test_name=ve_counters_1

./${test_name}

#!/bin/bash
set -x
REF_OUT_DIR=ref_output
REF_IN_DIR=ref_input

./test_vftrace vftr_symbols_test_1 $REF_IN_DIR/vftr_symbols_test_1.x
diff $REF_OUT_DIR/vftr_symbols_test_1.out vftr_symbols_test_1.out

# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_LINK_UNIT_TEST_FILES
#
# DESCRIPTION
#
#   This macro links the test files and scripts to the test folders
#

AC_DEFUN([AX_LINK_UNIT_TEST_FILES], [
   AC_PREREQ(2.50)
   # Test files and folders
AC_CONFIG_LINKS(test/units/ref_input/scenario_1.json:test/units/ref_input/scenario_1.json
                test/units/ref_input/scenario_2.json:test/units/ref_input/scenario_2.json
                test/units/ref_input/scenario_3.json:test/units/ref_input/scenario_3.json
                test/units/ref_input/symbols_test_1.x:test/units/ref_input/symbols_test_1.x
               )
AC_CONFIG_LINKS(test/units/ref_output/environment_1.out:test/units/ref_output/environment_1.out
                test/units/ref_output/environment_2.out:test/units/ref_output/environment_2.out
                test/units/ref_output/filewrite_1.out:test/units/ref_output/filewrite_1.out
                test/units/ref_output/scenario_1.out:test/units/ref_output/scenario_1.out
                test/units/ref_output/scenario_2.out:test/units/ref_output/scenario_2.out
                test/units/ref_output/setup_1.out:test/units/ref_output/setup_1.out
                test/units/ref_output/setup_2.out:test/units/ref_output/setup_2.out
                test/units/ref_output/symbols_test_1.out:test/units/ref_output/symbols_test_1.out
                test/units/ref_output/vftr_sxhwc_test_1.out:test/units/ref_output/vftr_sxhwc_test_1.out
                test/units/ref_output/vftr_browse_test_1.out:test/units/ref_output/vftr_browse_test_1.out
               )

AM_COND_IF([WITH_MPI],
           [units_refdir=mpi],
           [units_refdir=serial])

AC_CONFIG_LINKS(test/units/ref_output/functions_1.out:test/units/ref_output/$units_refdir/functions_1.out
                test/units/ref_output/functions_2.out:test/units/ref_output/$units_refdir/functions_2.out
                test/units/ref_output/functions_3.out:test/units/ref_output/$units_refdir/functions_3.out
                test/units/ref_output/functions_4.out:test/units/ref_output/$units_refdir/functions_4.out
                test/units/ref_output/functions_5.out:test/units/ref_output/$units_refdir/functions_5.out
                test/units/ref_output/filewrite_2.out:test/units/ref_output/$units_refdir/filewrite_2.out
                test/units/ref_output/stacks_1.out:test/units/ref_output/$units_refdir/stacks_1.out)

AM_COND_IF([WITH_MPI],
           [AC_CONFIG_LINKS(test/units/ref_output/vftr_stacks_test_2.out:test/units/ref_output/$units_refdir/vftr_stacks_test_2.out)]
          )

AC_CONFIG_LINKS(test/units/this_passes.sh:test/units/this_passes.sh
                test/units/this_fails.sh:test/units/this_fails.sh
                test/units/symbols_test_1.sh:test/units/symbols_test_1.sh
                test/units/environment_1.sh:test/units/environment_1.sh
                test/units/environment_2.sh:test/units/environment_2.sh
                test/units/setup_1.sh:test/units/setup_1.sh
                test/units/setup_2.sh:test/units/setup_2.sh
                test/units/filewrite_1.sh:test/units/filewrite_1.sh
                test/units/filewrite_2.sh:test/units/filewrite_2.sh
                test/units/scenario_1.sh:test/units/scenario_1.sh
                test/units/scenario_2.sh:test/units/scenario_2.sh
                test/units/scenario_3.sh:test/units/scenario_3.sh
                test/units/functions_1.sh:test/units/functions_1.sh
                test/units/functions_2.sh:test/units/functions_2.sh
                test/units/functions_3.sh:test/units/functions_3.sh
                test/units/functions_4.sh:test/units/functions_4.sh
                test/units/functions_5.sh:test/units/functions_5.sh
                test/units/stacks_1.sh:test/units/stacks_1.sh
                test/units/vftr_sxhwc_test_1.sh:test/units/vftr_sxhwc_test_1.sh
                test/units/vftr_browse_test_1.sh:test/units/vftr_browse_test_1.sh
               )

AM_COND_IF([WITH_MPI],
           [AC_CONFIG_LINKS(test/units/vftr_stacks_test_2.sh:test/units/vftr_stacks_test_2.sh)]
          )

AC_CONFIG_LINKS(test/units/radixsort_uint64.sh:test/units/radixsort_uint64.sh
                test/units/sort_integer_ascending.sh:test/units/sort_integer_ascending.sh
                test/units/sort_integer_descending.sh:test/units/sort_integer_descending.sh
                test/units/sort_double_ascending.sh:test/units/sort_double_ascending.sh
                test/units/sort_double_descending.sh:test/units/sort_double_descending.sh
                test/units/sort_double_copy_ascending.sh:test/units/sort_double_copy_ascending.sh
                test/units/sort_double_copy_descending.sh:test/units/sort_double_copy_descending.sh
               )
])

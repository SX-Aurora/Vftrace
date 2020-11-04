AC_INIT([vftrace], [1.0], [felix.uhl@emea.nec.com,christian.weiss@emea.nec.com])
: ${CFLAGS="-O2"}
: ${CXXFLAGS="-O2"}
: ${FCFLAGS="-O2"}

# Will be generated
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])
AM_INIT_AUTOMAKE([-Wall -Werror foreign no-dependencies subdir-objects])
AC_PROG_CC
AC_PROG_CXX
AC_PROG_FC
AC_FC_SRCEXT(F90)
AM_PROG_AR
#AC_PROG_RANLIB
# Enable per-target flags for C
# According to the documentation, this should be obsolete
# but without, autoreconf explicitly asks for it ... ?
AM_PROG_CC_C_O
# WO_PROG_COMPILERS()
################################
# Check if C-compiler supports c99 standard
AX_CHECK_C99_STD

# Check if elf-symbols align with function pointers
AX_CHECK_VMAP_OFFSET

# Check if on VE or VH (x86)
AX_CHECK_VE
################################

# Check MPI
AX_CHECK_MPI

# Enable hardware counters
AX_ENABLE_HWC

# Check for Papi library
AX_PAPI_LIBRARY

# Check consistency of HWC, Architecture, and Papi
AX_CHECK_HWC_CONSISTENCY

AC_MSG_CHECKING([if Intel compiler is used])
check_if_intel_compiler='$CC --version | grep Intel'
if AC_TRY_EVAL(check_if_intel_compiler); then
  uses_intel_compiler=yes
else
  uses_intel_compiler=no
fi 
AC_MSG_RESULT([$uses_intel_compiler])
AM_CONDITIONAL([USES_INTEL_COMPILER], [test "$uses_intel_compiler" = "yes"])

LT_INIT(dlopen)
AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([src/Makefile])
AC_CONFIG_FILES([src/vftr_mpi/Makefile])
AC_CONFIG_FILES([src/vftr_mpi/c-wrapper/Makefile])
AC_CONFIG_FILES([src/vftr_mpi/f-wrapper/Makefile])
AC_CONFIG_FILES([src/vftr_mpi/f08-wrapper/Makefile])
AC_CONFIG_FILES([tools/Makefile])
AC_CONFIG_FILES([test/Makefile])
AC_CONFIG_FILES([test/units/Makefile])
AC_CONFIG_FILES([test/user_functions/Makefile])
AC_CONFIG_FILES([test/mpi/Makefile])
AC_CONFIG_FILES([test/mpi/c-mpi/Makefile])
AC_CONFIG_FILES([test/mpi/f-mpi/Makefile])
AC_CONFIG_FILES([test/mpi/f08-mpi/Makefile])

dnl # # Test files and folders
dnl # AC_CONFIG_LINKS(test/units/ref_input/vftr_scenario_test_1.json:test/units/ref_input/vftr_scenario_test_1.json
dnl #                 test/units/ref_input/vftr_scenario_test_2.json:test/units/ref_input/vftr_scenario_test_2.json
dnl #                 test/units/ref_input/vftr_scenario_test_3.json:test/units/ref_input/vftr_scenario_test_3.json
dnl #                 test/units/ref_input/vftr_symbols_test_1.x:test/units/ref_input/vftr_symbols_test_1.x
dnl #                )
dnl # AC_CONFIG_LINKS(test/units/ref_output/vftr_environment_test_1.out:test/units/ref_output/vftr_environment_test_1.out
dnl #                 test/units/ref_output/vftr_filewrite_test_1.out:test/units/ref_output/vftr_filewrite_test_1.out
dnl #                 test/units/ref_output/vftr_scenario_test_1.out:test/units/ref_output/vftr_scenario_test_1.out
dnl #                 test/units/ref_output/vftr_scenario_test_2.out:test/units/ref_output/vftr_scenario_test_2.out
dnl #                 test/units/ref_output/vftr_setup_test_1.out:test/units/ref_output/vftr_setup_test_1.out
dnl #                 test/units/ref_output/vftr_setup_test_2.out:test/units/ref_output/vftr_setup_test_2.out
dnl #                 test/units/ref_output/vftr_symbols_test_1.out:test/units/ref_output/vftr_symbols_test_1.out
dnl #                 test/units/ref_output/vftr_sxhwc_test_1.out:test/units/ref_output/vftr_sxhwc_test_1.out
dnl #                )
dnl # 
dnl # AM_COND_IF([WITH_MPI],
dnl #            [units_refdir=mpi],
dnl #            [units_refdir=serial])
dnl # 
dnl # AC_CONFIG_LINKS(test/units/ref_output/vftr_functions_test_1.out:test/units/ref_output/$units_refdir/vftr_functions_test_1.out
dnl #                 test/units/ref_output/vftr_functions_test_2.out:test/units/ref_output/$units_refdir/vftr_functions_test_2.out
dnl #                 test/units/ref_output/vftr_functions_test_3.out:test/units/ref_output/$units_refdir/vftr_functions_test_3.out
dnl #                 test/units/ref_output/vftr_functions_test_4.out:test/units/ref_output/$units_refdir/vftr_functions_test_4.out
dnl #                 test/units/ref_output/vftr_functions_test_5.out:test/units/ref_output/$units_refdir/vftr_functions_test_5.out
dnl #                 test/units/ref_output/vftr_filewrite_test_2.out:test/units/ref_output/$units_refdir/vftr_filewrite_test_2.out
dnl #                 test/units/ref_output/vftr_stacks_test_1.out:test/units/ref_output/$units_refdir/vftr_stacks_test_1.out)
dnl # 
dnl # AM_COND_IF([WITH_MPI],
dnl #            [AC_CONFIG_LINKS(test/units/ref_output/vftr_stacks_test_2.out:test/units/ref_output/$units_refdir/vftr_stacks_test_2.out)]
dnl #           )
dnl # 
dnl # AC_CONFIG_LINKS(test/units/this_passes.sh:test/units/this_passes.sh
dnl #                 test/units/this_fails.sh:test/units/this_fails.sh
dnl #                 test/units/vftr_symbols_test_1.sh:test/units/vftr_symbols_test_1.sh
dnl #                 test/units/vftr_environment_test_1.sh:test/units/vftr_environment_test_1.sh
dnl #                 test/units/vftr_setup_test_1.sh:test/units/vftr_setup_test_1.sh
dnl #                 test/units/vftr_setup_test_2.sh:test/units/vftr_setup_test_2.sh
dnl #                 test/units/vftr_filewrite_test_1.sh:test/units/vftr_filewrite_test_1.sh
dnl #                 test/units/vftr_filewrite_test_2.sh:test/units/vftr_filewrite_test_2.sh
dnl #                 test/units/vftr_scenario_test_1.sh:test/units/vftr_scenario_test_1.sh
dnl #                 test/units/vftr_scenario_test_2.sh:test/units/vftr_scenario_test_2.sh
dnl #                 test/units/vftr_scenario_test_3.sh:test/units/vftr_scenario_test_3.sh
dnl #                 test/units/vftr_functions_test_1.sh:test/units/vftr_functions_test_1.sh
dnl #                 test/units/vftr_functions_test_2.sh:test/units/vftr_functions_test_2.sh
dnl #                 test/units/vftr_functions_test_3.sh:test/units/vftr_functions_test_3.sh
dnl #                 test/units/vftr_functions_test_4.sh:test/units/vftr_functions_test_4.sh
dnl #                 test/units/vftr_functions_test_5.sh:test/units/vftr_functions_test_5.sh
dnl #                 test/units/vftr_stacks_test_1.sh:test/units/vftr_stacks_test_1.sh
dnl #                 test/units/vftr_sxhwc_test_1.sh:test/units/vftr_sxhwc_test_1.sh
dnl #                )
dnl # AM_COND_IF([WITH_MPI],
dnl #            [AC_CONFIG_LINKS(test/units/vftr_stacks_test_2.sh:test/units/vftr_stacks_test_2.sh)]
dnl #           )
dnl # 
dnl # # User function tests
dnl # AC_CONFIG_LINKS(test/user_functions/cregions1.sh:test/user_functions/cregions1.sh
dnl #                 test/user_functions/cregions2.sh:test/user_functions/cregions2.sh
dnl #                 test/user_functions/cregions3.sh:test/user_functions/cregions3.sh
dnl #                 test/user_functions/cregions4.sh:test/user_functions/cregions4.sh
dnl #                 test/user_functions/fregions1.sh:test/user_functions/fregions1.sh
dnl #                 test/user_functions/fregions2.sh:test/user_functions/fregions2.sh
dnl #                 test/user_functions/fregions3.sh:test/user_functions/fregions3.sh
dnl #                 test/user_functions/fregions4.sh:test/user_functions/fregions4.sh
dnl #                )
dnl # AC_CONFIG_LINKS(test/user_functions/cget_stack.sh:test/user_functions/cget_stack.sh
dnl #                 test/user_functions/fget_stack.sh:test/user_functions/fget_stack.sh
dnl #                )
dnl # AC_CONFIG_LINKS(test/user_functions/cpause_resume.sh:test/user_functions/cpause_resume.sh
dnl #                 test/user_functions/fpause_resume.sh:test/user_functions/fpause_resume.sh
dnl #                )
dnl # 
dnl # # C-MPI testscripts
dnl # AM_COND_IF([WITH_MPI],
dnl #            [AC_CONFIG_LINKS(test/mpi/c-mpi/init_finalize-1.sh:test/mpi/testscripts/init_finalize-1.sh
dnl #                             test/mpi/c-mpi/init_finalize-2.sh:test/mpi/testscripts/init_finalize-2.sh
dnl #                             test/mpi/c-mpi/init_finalize-3.sh:test/mpi/testscripts/init_finalize-3.sh
dnl #                             test/mpi/c-mpi/init_finalize-4.sh:test/mpi/testscripts/init_finalize-4.sh
dnl #                             test/mpi/c-mpi/ctypes.sh:test/mpi/testscripts/ctypes.sh
dnl #                             test/mpi/c-mpi/derived_ctypes.sh:test/mpi/testscripts/derived_ctypes.sh
dnl #                             test/mpi/c-mpi/pcontrol.sh:test/mpi/testscripts/pcontrol.sh
dnl #                             test/mpi/c-mpi/send_recv.sh:test/mpi/testscripts/send_recv.sh
dnl #                             test/mpi/c-mpi/isend_recv.sh:test/mpi/testscripts/isend_recv.sh
dnl #                             test/mpi/c-mpi/bsend_recv.sh:test/mpi/testscripts/bsend_recv.sh
dnl #                             test/mpi/c-mpi/ibsend_recv.sh:test/mpi/testscripts/ibsend_recv.sh
dnl #                             test/mpi/c-mpi/ssend_recv.sh:test/mpi/testscripts/ssend_recv.sh
dnl #                             test/mpi/c-mpi/issend_recv.sh:test/mpi/testscripts/issend_recv.sh
dnl #                             test/mpi/c-mpi/rsend_recv.sh:test/mpi/testscripts/rsend_recv.sh
dnl #                             test/mpi/c-mpi/irsend_recv.sh:test/mpi/testscripts/irsend_recv.sh
dnl #                             test/mpi/c-mpi/send_irecv.sh:test/mpi/testscripts/send_irecv.sh
dnl #                             test/mpi/c-mpi/sendrecv.sh:test/mpi/testscripts/sendrecv.sh
dnl #                             test/mpi/c-mpi/sendrecv_replace.sh:test/mpi/testscripts/sendrecv_replace.sh
dnl #                             test/mpi/c-mpi/probe.sh:test/mpi/testscripts/probe.sh
dnl #                             test/mpi/c-mpi/iprobe.sh:test/mpi/testscripts/iprobe.sh
dnl #                             test/mpi/c-mpi/start.sh:test/mpi/testscripts/start.sh
dnl #                             test/mpi/c-mpi/startall.sh:test/mpi/testscripts/startall.sh
dnl #                             test/mpi/c-mpi/send_init.sh:test/mpi/testscripts/send_init.sh
dnl #                             test/mpi/c-mpi/bsend_init.sh:test/mpi/testscripts/bsend_init.sh
dnl #                             test/mpi/c-mpi/ssend_init.sh:test/mpi/testscripts/ssend_init.sh
dnl #                             test/mpi/c-mpi/rsend_init.sh:test/mpi/testscripts/rsend_init.sh
dnl #                             test/mpi/c-mpi/recv_init.sh:test/mpi/testscripts/recv_init.sh
dnl #                             test/mpi/c-mpi/barrier.sh:test/mpi/testscripts/barrier.sh
dnl #                             test/mpi/c-mpi/bcast.sh:test/mpi/testscripts/bcast.sh
dnl #                             test/mpi/c-mpi/bcast_intercom.sh:test/mpi/testscripts/bcast_intercom.sh
dnl #                             test/mpi/c-mpi/ibcast.sh:test/mpi/testscripts/ibcast.sh
dnl #                             test/mpi/c-mpi/ibcast_intercom.sh:test/mpi/testscripts/ibcast_intercom.sh
dnl #                             test/mpi/c-mpi/gather.sh:test/mpi/testscripts/gather.sh
dnl #                             test/mpi/c-mpi/gather_intercom.sh:test/mpi/testscripts/gather_intercom.sh
dnl #                             test/mpi/c-mpi/gather_inplace.sh:test/mpi/testscripts/gather_inplace.sh
dnl #                             test/mpi/c-mpi/igather.sh:test/mpi/testscripts/igather.sh
dnl #                             test/mpi/c-mpi/igather_intercom.sh:test/mpi/testscripts/igather_intercom.sh
dnl #                             test/mpi/c-mpi/igather_inplace.sh:test/mpi/testscripts/igather_inplace.sh
dnl #                             test/mpi/c-mpi/gatherv.sh:test/mpi/testscripts/gatherv.sh
dnl #                             test/mpi/c-mpi/gatherv_intercom.sh:test/mpi/testscripts/gatherv_intercom.sh
dnl #                             test/mpi/c-mpi/gatherv_inplace.sh:test/mpi/testscripts/gatherv_inplace.sh
dnl #                             test/mpi/c-mpi/igatherv.sh:test/mpi/testscripts/igatherv.sh
dnl #                             test/mpi/c-mpi/igatherv_intercom.sh:test/mpi/testscripts/igatherv_intercom.sh
dnl #                             test/mpi/c-mpi/igatherv_inplace.sh:test/mpi/testscripts/igatherv_inplace.sh
dnl #                             test/mpi/c-mpi/scatter.sh:test/mpi/testscripts/scatter.sh
dnl #                             test/mpi/c-mpi/scatter_intercom.sh:test/mpi/testscripts/scatter_intercom.sh
dnl #                             test/mpi/c-mpi/scatter_inplace.sh:test/mpi/testscripts/scatter_inplace.sh
dnl #                             test/mpi/c-mpi/iscatter.sh:test/mpi/testscripts/iscatter.sh
dnl #                             test/mpi/c-mpi/iscatter_intercom.sh:test/mpi/testscripts/iscatter_intercom.sh
dnl #                             test/mpi/c-mpi/iscatter_inplace.sh:test/mpi/testscripts/iscatter_inplace.sh
dnl #                             test/mpi/c-mpi/scatterv.sh:test/mpi/testscripts/scatterv.sh
dnl #                             test/mpi/c-mpi/scatterv_intercom.sh:test/mpi/testscripts/scatterv_intercom.sh
dnl #                             test/mpi/c-mpi/scatterv_inplace.sh:test/mpi/testscripts/scatterv_inplace.sh
dnl #                             test/mpi/c-mpi/iscatterv.sh:test/mpi/testscripts/iscatterv.sh
dnl #                             test/mpi/c-mpi/iscatterv_intercom.sh:test/mpi/testscripts/iscatterv_intercom.sh
dnl #                             test/mpi/c-mpi/iscatterv_inplace.sh:test/mpi/testscripts/iscatterv_inplace.sh
dnl #                             test/mpi/c-mpi/allgather.sh:test/mpi/testscripts/allgather.sh
dnl #                             test/mpi/c-mpi/allgather_intercom.sh:test/mpi/testscripts/allgather_intercom.sh
dnl #                             test/mpi/c-mpi/allgather_inplace.sh:test/mpi/testscripts/allgather_inplace.sh
dnl #                             test/mpi/c-mpi/iallgather.sh:test/mpi/testscripts/iallgather.sh
dnl #                             test/mpi/c-mpi/iallgather_intercom.sh:test/mpi/testscripts/iallgather_intercom.sh
dnl #                             test/mpi/c-mpi/iallgather_inplace.sh:test/mpi/testscripts/iallgather_inplace.sh
dnl #                             test/mpi/c-mpi/allgatherv.sh:test/mpi/testscripts/allgatherv.sh
dnl #                             test/mpi/c-mpi/allgatherv_intercom.sh:test/mpi/testscripts/allgatherv_intercom.sh
dnl #                             test/mpi/c-mpi/allgatherv_inplace.sh:test/mpi/testscripts/allgatherv_inplace.sh
dnl #                             test/mpi/c-mpi/iallgatherv.sh:test/mpi/testscripts/iallgatherv.sh
dnl #                             test/mpi/c-mpi/iallgatherv_intercom.sh:test/mpi/testscripts/iallgatherv_intercom.sh
dnl #                             test/mpi/c-mpi/iallgatherv_inplace.sh:test/mpi/testscripts/iallgatherv_inplace.sh
dnl #                             test/mpi/c-mpi/alltoall.sh:test/mpi/testscripts/alltoall.sh
dnl #                             test/mpi/c-mpi/alltoall_intercom.sh:test/mpi/testscripts/alltoall_intercom.sh
dnl #                             test/mpi/c-mpi/alltoall_inplace.sh:test/mpi/testscripts/alltoall_inplace.sh
dnl #                             test/mpi/c-mpi/ialltoall.sh:test/mpi/testscripts/ialltoall.sh
dnl #                             test/mpi/c-mpi/ialltoall_intercom.sh:test/mpi/testscripts/ialltoall_intercom.sh
dnl #                             test/mpi/c-mpi/ialltoall_inplace.sh:test/mpi/testscripts/ialltoall_inplace.sh
dnl #                             test/mpi/c-mpi/alltoallv.sh:test/mpi/testscripts/alltoallv.sh
dnl #                             test/mpi/c-mpi/alltoallv_intercom.sh:test/mpi/testscripts/alltoallv_intercom.sh
dnl #                             test/mpi/c-mpi/alltoallv_inplace.sh:test/mpi/testscripts/alltoallv_inplace.sh
dnl #                             test/mpi/c-mpi/ialltoallv.sh:test/mpi/testscripts/ialltoallv.sh
dnl #                             test/mpi/c-mpi/ialltoallv_intercom.sh:test/mpi/testscripts/ialltoallv_intercom.sh
dnl #                             test/mpi/c-mpi/ialltoallv_inplace.sh:test/mpi/testscripts/ialltoallv_inplace.sh
dnl #                             test/mpi/c-mpi/alltoallw.sh:test/mpi/testscripts/alltoallw.sh
dnl #                             test/mpi/c-mpi/alltoallw_intercom.sh:test/mpi/testscripts/alltoallw_intercom.sh
dnl #                             test/mpi/c-mpi/alltoallw_inplace.sh:test/mpi/testscripts/alltoallw_inplace.sh
dnl #                             test/mpi/c-mpi/ialltoallw.sh:test/mpi/testscripts/ialltoallw.sh
dnl #                             test/mpi/c-mpi/ialltoallw_intercom.sh:test/mpi/testscripts/ialltoallw_intercom.sh
dnl #                             test/mpi/c-mpi/ialltoallw_inplace.sh:test/mpi/testscripts/ialltoallw_inplace.sh
dnl #                             test/mpi/c-mpi/reduce.sh:test/mpi/testscripts/reduce.sh
dnl #                             test/mpi/c-mpi/reduce_intercom.sh:test/mpi/testscripts/reduce_intercom.sh
dnl #                             test/mpi/c-mpi/reduce_inplace.sh:test/mpi/testscripts/reduce_inplace.sh
dnl #                             test/mpi/c-mpi/ireduce.sh:test/mpi/testscripts/ireduce.sh
dnl #                             test/mpi/c-mpi/ireduce_intercom.sh:test/mpi/testscripts/ireduce_intercom.sh
dnl #                             test/mpi/c-mpi/ireduce_inplace.sh:test/mpi/testscripts/ireduce_inplace.sh
dnl #                             test/mpi/c-mpi/allreduce.sh:test/mpi/testscripts/allreduce.sh
dnl #                             test/mpi/c-mpi/allreduce_intercom.sh:test/mpi/testscripts/allreduce_intercom.sh
dnl #                             test/mpi/c-mpi/allreduce_inplace.sh:test/mpi/testscripts/allreduce_inplace.sh
dnl #                             test/mpi/c-mpi/iallreduce.sh:test/mpi/testscripts/iallreduce.sh
dnl #                             test/mpi/c-mpi/iallreduce_intercom.sh:test/mpi/testscripts/iallreduce_intercom.sh
dnl #                             test/mpi/c-mpi/iallreduce_inplace.sh:test/mpi/testscripts/iallreduce_inplace.sh
dnl #                             test/mpi/c-mpi/reduce_scatter_block.sh:test/mpi/testscripts/reduce_scatter_block.sh
dnl #                             test/mpi/c-mpi/reduce_scatter_block_intercom.sh:test/mpi/testscripts/reduce_scatter_block_intercom.sh
dnl #                             test/mpi/c-mpi/reduce_scatter_block_inplace.sh:test/mpi/testscripts/reduce_scatter_block_inplace.sh
dnl #                             test/mpi/c-mpi/ireduce_scatter_block.sh:test/mpi/testscripts/ireduce_scatter_block.sh
dnl #                             test/mpi/c-mpi/ireduce_scatter_block_intercom.sh:test/mpi/testscripts/ireduce_scatter_block_intercom.sh
dnl #                             test/mpi/c-mpi/ireduce_scatter_block_inplace.sh:test/mpi/testscripts/ireduce_scatter_block_inplace.sh
dnl #                             test/mpi/c-mpi/reduce_scatter.sh:test/mpi/testscripts/reduce_scatter.sh
dnl #                             test/mpi/c-mpi/reduce_scatter_intercom.sh:test/mpi/testscripts/reduce_scatter_intercom.sh
dnl #                             test/mpi/c-mpi/reduce_scatter_inplace.sh:test/mpi/testscripts/reduce_scatter_inplace.sh
dnl #                             test/mpi/c-mpi/ireduce_scatter.sh:test/mpi/testscripts/ireduce_scatter.sh
dnl #                             test/mpi/c-mpi/ireduce_scatter_intercom.sh:test/mpi/testscripts/ireduce_scatter_intercom.sh
dnl #                             test/mpi/c-mpi/ireduce_scatter_inplace.sh:test/mpi/testscripts/ireduce_scatter_inplace.sh
dnl #                             test/mpi/c-mpi/scan.sh:test/mpi/testscripts/scan.sh
dnl #                             test/mpi/c-mpi/scan_inplace.sh:test/mpi/testscripts/scan_inplace.sh
dnl #                             test/mpi/c-mpi/iscan.sh:test/mpi/testscripts/iscan.sh
dnl #                             test/mpi/c-mpi/iscan_inplace.sh:test/mpi/testscripts/iscan_inplace.sh
dnl #                             test/mpi/c-mpi/exscan.sh:test/mpi/testscripts/exscan.sh
dnl #                             test/mpi/c-mpi/exscan_inplace.sh:test/mpi/testscripts/exscan_inplace.sh
dnl #                             test/mpi/c-mpi/iexscan.sh:test/mpi/testscripts/iexscan.sh
dnl #                             test/mpi/c-mpi/iexscan_inplace.sh:test/mpi/testscripts/iexscan_inplace.sh
dnl #                             test/mpi/c-mpi/put.sh:test/mpi/testscripts/put.sh
dnl #                             test/mpi/c-mpi/rput.sh:test/mpi/testscripts/rput.sh
dnl #                             test/mpi/c-mpi/get.sh:test/mpi/testscripts/get.sh
dnl #                             test/mpi/c-mpi/rget.sh:test/mpi/testscripts/rget.sh
dnl #                             test/mpi/c-mpi/accumulate.sh:test/mpi/testscripts/accumulate.sh
dnl #                             test/mpi/c-mpi/raccumulate.sh:test/mpi/testscripts/raccumulate.sh
dnl #                             test/mpi/c-mpi/get_accumulate.sh:test/mpi/testscripts/get_accumulate.sh
dnl #                             test/mpi/c-mpi/rget_accumulate.sh:test/mpi/testscripts/rget_accumulate.sh
dnl #                             test/mpi/c-mpi/fetch_and_op.sh:test/mpi/testscripts/fetch_and_op.sh
dnl #                             test/mpi/c-mpi/compare_and_swap.sh:test/mpi/testscripts/compare_and_swap.sh
dnl #                             test/mpi/c-mpi/test.sh:test/mpi/testscripts/test.sh
dnl #                             test/mpi/c-mpi/testany.sh:test/mpi/testscripts/testany.sh
dnl #                             test/mpi/c-mpi/testsome.sh:test/mpi/testscripts/testsome.sh
dnl #                             test/mpi/c-mpi/testall.sh:test/mpi/testscripts/testall.sh
dnl #                             test/mpi/c-mpi/wait.sh:test/mpi/testscripts/wait.sh
dnl #                             test/mpi/c-mpi/waitany.sh:test/mpi/testscripts/waitany.sh
dnl #                             test/mpi/c-mpi/waitsome.sh:test/mpi/testscripts/waitsome.sh
dnl #                             test/mpi/c-mpi/waitall.sh:test/mpi/testscripts/waitall.sh
dnl #                            )
dnl #            ])
dnl #                 
dnl # # F-MPI testscripts
dnl # AM_COND_IF([WITH_MPI],
dnl #            [AC_CONFIG_LINKS(test/mpi/f-mpi/init_finalize-1.sh:test/mpi/testscripts/init_finalize-1.sh
dnl #                             test/mpi/f-mpi/init_finalize-2.sh:test/mpi/testscripts/init_finalize-2.sh
dnl #                             test/mpi/f-mpi/init_finalize-3.sh:test/mpi/testscripts/init_finalize-3.sh
dnl #                             test/mpi/f-mpi/init_finalize-4.sh:test/mpi/testscripts/init_finalize-4.sh
dnl #                             test/mpi/f-mpi/ftypes.sh:test/mpi/testscripts/ftypes.sh
dnl #                             test/mpi/f-mpi/derived_ftypes.sh:test/mpi/testscripts/derived_ftypes.sh
dnl #                             test/mpi/f-mpi/pcontrol.sh:test/mpi/testscripts/pcontrol.sh
dnl #                             test/mpi/f-mpi/send_recv.sh:test/mpi/testscripts/send_recv.sh
dnl #                             test/mpi/f-mpi/isend_recv.sh:test/mpi/testscripts/isend_recv.sh
dnl #                             test/mpi/f-mpi/bsend_recv.sh:test/mpi/testscripts/bsend_recv.sh
dnl #                             test/mpi/f-mpi/ibsend_recv.sh:test/mpi/testscripts/ibsend_recv.sh
dnl #                             test/mpi/f-mpi/ssend_recv.sh:test/mpi/testscripts/ssend_recv.sh
dnl #                             test/mpi/f-mpi/issend_recv.sh:test/mpi/testscripts/issend_recv.sh
dnl #                             test/mpi/f-mpi/rsend_recv.sh:test/mpi/testscripts/rsend_recv.sh
dnl #                             test/mpi/f-mpi/irsend_recv.sh:test/mpi/testscripts/irsend_recv.sh
dnl #                             test/mpi/f-mpi/send_irecv.sh:test/mpi/testscripts/send_irecv.sh
dnl #                             test/mpi/f-mpi/sendrecv.sh:test/mpi/testscripts/sendrecv.sh
dnl #                             test/mpi/f-mpi/sendrecv_replace.sh:test/mpi/testscripts/sendrecv_replace.sh
dnl #                             test/mpi/f-mpi/probe.sh:test/mpi/testscripts/probe.sh
dnl #                             test/mpi/f-mpi/iprobe.sh:test/mpi/testscripts/iprobe.sh
dnl #                             test/mpi/f-mpi/start.sh:test/mpi/testscripts/start.sh
dnl #                             test/mpi/f-mpi/startall.sh:test/mpi/testscripts/startall.sh
dnl #                             test/mpi/f-mpi/send_init.sh:test/mpi/testscripts/send_init.sh
dnl #                             test/mpi/f-mpi/bsend_init.sh:test/mpi/testscripts/bsend_init.sh
dnl #                             test/mpi/f-mpi/ssend_init.sh:test/mpi/testscripts/ssend_init.sh
dnl #                             test/mpi/f-mpi/rsend_init.sh:test/mpi/testscripts/rsend_init.sh
dnl #                             test/mpi/f-mpi/recv_init.sh:test/mpi/testscripts/recv_init.sh
dnl #                             test/mpi/f-mpi/barrier.sh:test/mpi/testscripts/barrier.sh
dnl #                             test/mpi/f-mpi/bcast.sh:test/mpi/testscripts/bcast.sh
dnl #                             test/mpi/f-mpi/bcast_intercom.sh:test/mpi/testscripts/bcast_intercom.sh
dnl #                             test/mpi/f-mpi/ibcast.sh:test/mpi/testscripts/ibcast.sh
dnl #                             test/mpi/f-mpi/ibcast_intercom.sh:test/mpi/testscripts/ibcast_intercom.sh
dnl #                             test/mpi/f-mpi/gather.sh:test/mpi/testscripts/gather.sh
dnl #                             test/mpi/f-mpi/gather_intercom.sh:test/mpi/testscripts/gather_intercom.sh
dnl #                             test/mpi/f-mpi/gather_inplace.sh:test/mpi/testscripts/gather_inplace.sh
dnl #                             test/mpi/f-mpi/igather.sh:test/mpi/testscripts/igather.sh
dnl #                             test/mpi/f-mpi/igather_intercom.sh:test/mpi/testscripts/igather_intercom.sh
dnl #                             test/mpi/f-mpi/igather_inplace.sh:test/mpi/testscripts/igather_inplace.sh
dnl #                             test/mpi/f-mpi/gatherv.sh:test/mpi/testscripts/gatherv.sh
dnl #                             test/mpi/f-mpi/gatherv_intercom.sh:test/mpi/testscripts/gatherv_intercom.sh
dnl #                             test/mpi/f-mpi/gatherv_inplace.sh:test/mpi/testscripts/gatherv_inplace.sh
dnl #                             test/mpi/f-mpi/igatherv.sh:test/mpi/testscripts/igatherv.sh
dnl #                             test/mpi/f-mpi/igatherv_intercom.sh:test/mpi/testscripts/igatherv_intercom.sh
dnl #                             test/mpi/f-mpi/igatherv_inplace.sh:test/mpi/testscripts/igatherv_inplace.sh
dnl #                             test/mpi/f-mpi/scatter.sh:test/mpi/testscripts/scatter.sh
dnl #                             test/mpi/f-mpi/scatter_intercom.sh:test/mpi/testscripts/scatter_intercom.sh
dnl #                             test/mpi/f-mpi/scatter_inplace.sh:test/mpi/testscripts/scatter_inplace.sh
dnl #                             test/mpi/f-mpi/iscatter.sh:test/mpi/testscripts/iscatter.sh
dnl #                             test/mpi/f-mpi/iscatter_intercom.sh:test/mpi/testscripts/iscatter_intercom.sh
dnl #                             test/mpi/f-mpi/iscatter_inplace.sh:test/mpi/testscripts/iscatter_inplace.sh
dnl #                             test/mpi/f-mpi/scatterv.sh:test/mpi/testscripts/scatterv.sh
dnl #                             test/mpi/f-mpi/scatterv_intercom.sh:test/mpi/testscripts/scatterv_intercom.sh
dnl #                             test/mpi/f-mpi/scatterv_inplace.sh:test/mpi/testscripts/scatterv_inplace.sh
dnl #                             test/mpi/f-mpi/iscatterv.sh:test/mpi/testscripts/iscatterv.sh
dnl #                             test/mpi/f-mpi/iscatterv_intercom.sh:test/mpi/testscripts/iscatterv_intercom.sh
dnl #                             test/mpi/f-mpi/iscatterv_inplace.sh:test/mpi/testscripts/iscatterv_inplace.sh
dnl #                             test/mpi/f-mpi/allgather.sh:test/mpi/testscripts/allgather.sh
dnl #                             test/mpi/f-mpi/allgather_intercom.sh:test/mpi/testscripts/allgather_intercom.sh
dnl #                             test/mpi/f-mpi/allgather_inplace.sh:test/mpi/testscripts/allgather_inplace.sh
dnl #                             test/mpi/f-mpi/iallgather.sh:test/mpi/testscripts/iallgather.sh
dnl #                             test/mpi/f-mpi/iallgather_intercom.sh:test/mpi/testscripts/iallgather_intercom.sh
dnl #                             test/mpi/f-mpi/iallgather_inplace.sh:test/mpi/testscripts/iallgather_inplace.sh
dnl #                             test/mpi/f-mpi/allgatherv.sh:test/mpi/testscripts/allgatherv.sh
dnl #                             test/mpi/f-mpi/allgatherv_intercom.sh:test/mpi/testscripts/allgatherv_intercom.sh
dnl #                             test/mpi/f-mpi/allgatherv_inplace.sh:test/mpi/testscripts/allgatherv_inplace.sh
dnl #                             test/mpi/f-mpi/iallgatherv.sh:test/mpi/testscripts/iallgatherv.sh
dnl #                             test/mpi/f-mpi/iallgatherv_intercom.sh:test/mpi/testscripts/iallgatherv_intercom.sh
dnl #                             test/mpi/f-mpi/iallgatherv_inplace.sh:test/mpi/testscripts/iallgatherv_inplace.sh
dnl #                             test/mpi/f-mpi/alltoall.sh:test/mpi/testscripts/alltoall.sh
dnl #                             test/mpi/f-mpi/alltoall_intercom.sh:test/mpi/testscripts/alltoall_intercom.sh
dnl #                             test/mpi/f-mpi/alltoall_inplace.sh:test/mpi/testscripts/alltoall_inplace.sh
dnl #                             test/mpi/f-mpi/ialltoall.sh:test/mpi/testscripts/ialltoall.sh
dnl #                             test/mpi/f-mpi/ialltoall_intercom.sh:test/mpi/testscripts/ialltoall_intercom.sh
dnl #                             test/mpi/f-mpi/ialltoall_inplace.sh:test/mpi/testscripts/ialltoall_inplace.sh
dnl #                             test/mpi/f-mpi/alltoallv.sh:test/mpi/testscripts/alltoallv.sh
dnl #                             test/mpi/f-mpi/alltoallv_intercom.sh:test/mpi/testscripts/alltoallv_intercom.sh
dnl #                             test/mpi/f-mpi/alltoallv_inplace.sh:test/mpi/testscripts/alltoallv_inplace.sh
dnl #                             test/mpi/f-mpi/ialltoallv.sh:test/mpi/testscripts/ialltoallv.sh
dnl #                             test/mpi/f-mpi/ialltoallv_intercom.sh:test/mpi/testscripts/ialltoallv_intercom.sh
dnl #                             test/mpi/f-mpi/ialltoallv_inplace.sh:test/mpi/testscripts/ialltoallv_inplace.sh
dnl #                             test/mpi/f-mpi/alltoallw.sh:test/mpi/testscripts/alltoallw.sh
dnl #                             test/mpi/f-mpi/alltoallw_intercom.sh:test/mpi/testscripts/alltoallw_intercom.sh
dnl #                             test/mpi/f-mpi/alltoallw_inplace.sh:test/mpi/testscripts/alltoallw_inplace.sh
dnl #                             test/mpi/f-mpi/ialltoallw.sh:test/mpi/testscripts/ialltoallw.sh
dnl #                             test/mpi/f-mpi/ialltoallw_intercom.sh:test/mpi/testscripts/ialltoallw_intercom.sh
dnl #                             test/mpi/f-mpi/ialltoallw_inplace.sh:test/mpi/testscripts/ialltoallw_inplace.sh
dnl #                             test/mpi/f-mpi/reduce.sh:test/mpi/testscripts/reduce.sh
dnl #                             test/mpi/f-mpi/reduce_intercom.sh:test/mpi/testscripts/reduce_intercom.sh
dnl #                             test/mpi/f-mpi/reduce_inplace.sh:test/mpi/testscripts/reduce_inplace.sh
dnl #                             test/mpi/f-mpi/ireduce.sh:test/mpi/testscripts/ireduce.sh
dnl #                             test/mpi/f-mpi/ireduce_intercom.sh:test/mpi/testscripts/ireduce_intercom.sh
dnl #                             test/mpi/f-mpi/ireduce_inplace.sh:test/mpi/testscripts/ireduce_inplace.sh
dnl #                             test/mpi/f-mpi/allreduce.sh:test/mpi/testscripts/allreduce.sh
dnl #                             test/mpi/f-mpi/allreduce_intercom.sh:test/mpi/testscripts/allreduce_intercom.sh
dnl #                             test/mpi/f-mpi/allreduce_inplace.sh:test/mpi/testscripts/allreduce_inplace.sh
dnl #                             test/mpi/f-mpi/iallreduce.sh:test/mpi/testscripts/iallreduce.sh
dnl #                             test/mpi/f-mpi/iallreduce_intercom.sh:test/mpi/testscripts/iallreduce_intercom.sh
dnl #                             test/mpi/f-mpi/iallreduce_inplace.sh:test/mpi/testscripts/iallreduce_inplace.sh
dnl #                             test/mpi/f-mpi/reduce_scatter_block.sh:test/mpi/testscripts/reduce_scatter_block.sh
dnl #                             test/mpi/f-mpi/reduce_scatter_block_intercom.sh:test/mpi/testscripts/reduce_scatter_block_intercom.sh
dnl #                             test/mpi/f-mpi/reduce_scatter_block_inplace.sh:test/mpi/testscripts/reduce_scatter_block_inplace.sh
dnl #                             test/mpi/f-mpi/ireduce_scatter_block.sh:test/mpi/testscripts/ireduce_scatter_block.sh
dnl #                             test/mpi/f-mpi/ireduce_scatter_block_intercom.sh:test/mpi/testscripts/ireduce_scatter_block_intercom.sh
dnl #                             test/mpi/f-mpi/ireduce_scatter_block_inplace.sh:test/mpi/testscripts/ireduce_scatter_block_inplace.sh
dnl #                             test/mpi/f-mpi/reduce_scatter.sh:test/mpi/testscripts/reduce_scatter.sh
dnl #                             test/mpi/f-mpi/reduce_scatter_intercom.sh:test/mpi/testscripts/reduce_scatter_intercom.sh
dnl #                             test/mpi/f-mpi/reduce_scatter_inplace.sh:test/mpi/testscripts/reduce_scatter_inplace.sh
dnl #                             test/mpi/f-mpi/ireduce_scatter.sh:test/mpi/testscripts/ireduce_scatter.sh
dnl #                             test/mpi/f-mpi/ireduce_scatter_intercom.sh:test/mpi/testscripts/ireduce_scatter_intercom.sh
dnl #                             test/mpi/f-mpi/ireduce_scatter_inplace.sh:test/mpi/testscripts/ireduce_scatter_inplace.sh
dnl #                             test/mpi/f-mpi/scan.sh:test/mpi/testscripts/scan.sh
dnl #                             test/mpi/f-mpi/scan_inplace.sh:test/mpi/testscripts/scan_inplace.sh
dnl #                             test/mpi/f-mpi/iscan.sh:test/mpi/testscripts/iscan.sh
dnl #                             test/mpi/f-mpi/iscan_inplace.sh:test/mpi/testscripts/iscan_inplace.sh
dnl #                             test/mpi/f-mpi/exscan.sh:test/mpi/testscripts/exscan.sh
dnl #                             test/mpi/f-mpi/exscan_inplace.sh:test/mpi/testscripts/exscan_inplace.sh
dnl #                             test/mpi/f-mpi/iexscan.sh:test/mpi/testscripts/iexscan.sh
dnl #                             test/mpi/f-mpi/iexscan_inplace.sh:test/mpi/testscripts/iexscan_inplace.sh
dnl #                             test/mpi/f-mpi/put.sh:test/mpi/testscripts/put.sh
dnl #                             test/mpi/f-mpi/rput.sh:test/mpi/testscripts/rput.sh
dnl #                             test/mpi/f-mpi/get.sh:test/mpi/testscripts/get.sh
dnl #                             test/mpi/f-mpi/rget.sh:test/mpi/testscripts/rget.sh
dnl #                             test/mpi/f-mpi/accumulate.sh:test/mpi/testscripts/accumulate.sh
dnl #                             test/mpi/f-mpi/raccumulate.sh:test/mpi/testscripts/raccumulate.sh
dnl #                             test/mpi/f-mpi/get_accumulate.sh:test/mpi/testscripts/get_accumulate.sh
dnl #                             test/mpi/f-mpi/rget_accumulate.sh:test/mpi/testscripts/rget_accumulate.sh
dnl #                             test/mpi/f-mpi/fetch_and_op.sh:test/mpi/testscripts/fetch_and_op.sh
dnl #                             test/mpi/f-mpi/compare_and_swap.sh:test/mpi/testscripts/compare_and_swap.sh
dnl #                             test/mpi/f-mpi/test.sh:test/mpi/testscripts/test.sh
dnl #                             test/mpi/f-mpi/testany.sh:test/mpi/testscripts/testany.sh
dnl #                             test/mpi/f-mpi/testsome.sh:test/mpi/testscripts/testsome.sh
dnl #                             test/mpi/f-mpi/testall.sh:test/mpi/testscripts/testall.sh
dnl #                             test/mpi/f-mpi/wait.sh:test/mpi/testscripts/wait.sh
dnl #                             test/mpi/f-mpi/waitany.sh:test/mpi/testscripts/waitany.sh
dnl #                             test/mpi/f-mpi/waitsome.sh:test/mpi/testscripts/waitsome.sh
dnl #                             test/mpi/f-mpi/waitall.sh:test/mpi/testscripts/waitall.sh
dnl #                            )
dnl #            ])
dnl #                                
dnl # # F08-MPI testscripts
dnl # AM_COND_IF([WITH_MPI],
dnl #            [AC_CONFIG_LINKS(test/mpi/f08-mpi/init_finalize-1.sh:test/mpi/testscripts/init_finalize-1.sh
dnl #                             test/mpi/f08-mpi/init_finalize-2.sh:test/mpi/testscripts/init_finalize-2.sh
dnl #                             test/mpi/f08-mpi/init_finalize-3.sh:test/mpi/testscripts/init_finalize-3.sh
dnl #                             test/mpi/f08-mpi/init_finalize-4.sh:test/mpi/testscripts/init_finalize-4.sh
dnl #                             test/mpi/f08-mpi/ftypes.sh:test/mpi/testscripts/ftypes.sh
dnl #                             test/mpi/f08-mpi/derived_ftypes.sh:test/mpi/testscripts/derived_ftypes.sh
dnl #                             test/mpi/f08-mpi/pcontrol.sh:test/mpi/testscripts/pcontrol.sh
dnl #                             test/mpi/f08-mpi/send_recv.sh:test/mpi/testscripts/send_recv.sh
dnl #                             test/mpi/f08-mpi/isend_recv.sh:test/mpi/testscripts/isend_recv.sh
dnl #                             test/mpi/f08-mpi/bsend_recv.sh:test/mpi/testscripts/bsend_recv.sh
dnl #                             test/mpi/f08-mpi/ibsend_recv.sh:test/mpi/testscripts/ibsend_recv.sh
dnl #                             test/mpi/f08-mpi/ssend_recv.sh:test/mpi/testscripts/ssend_recv.sh
dnl #                             test/mpi/f08-mpi/issend_recv.sh:test/mpi/testscripts/issend_recv.sh
dnl #                             test/mpi/f08-mpi/rsend_recv.sh:test/mpi/testscripts/rsend_recv.sh
dnl #                             test/mpi/f08-mpi/irsend_recv.sh:test/mpi/testscripts/irsend_recv.sh
dnl #                             test/mpi/f08-mpi/send_irecv.sh:test/mpi/testscripts/send_irecv.sh
dnl #                             test/mpi/f08-mpi/sendrecv.sh:test/mpi/testscripts/sendrecv.sh
dnl #                             test/mpi/f08-mpi/sendrecv_replace.sh:test/mpi/testscripts/sendrecv_replace.sh
dnl #                             test/mpi/f08-mpi/probe.sh:test/mpi/testscripts/probe.sh
dnl #                             test/mpi/f08-mpi/iprobe.sh:test/mpi/testscripts/iprobe.sh
dnl #                             test/mpi/f08-mpi/start.sh:test/mpi/testscripts/start.sh
dnl #                             test/mpi/f08-mpi/startall.sh:test/mpi/testscripts/startall.sh
dnl #                             test/mpi/f08-mpi/send_init.sh:test/mpi/testscripts/send_init.sh
dnl #                             test/mpi/f08-mpi/bsend_init.sh:test/mpi/testscripts/bsend_init.sh
dnl #                             test/mpi/f08-mpi/ssend_init.sh:test/mpi/testscripts/ssend_init.sh
dnl #                             test/mpi/f08-mpi/rsend_init.sh:test/mpi/testscripts/rsend_init.sh
dnl #                             test/mpi/f08-mpi/recv_init.sh:test/mpi/testscripts/recv_init.sh
dnl #                             test/mpi/f08-mpi/barrier.sh:test/mpi/testscripts/barrier.sh
dnl #                             test/mpi/f08-mpi/bcast.sh:test/mpi/testscripts/bcast.sh
dnl #                             test/mpi/f08-mpi/bcast_intercom.sh:test/mpi/testscripts/bcast_intercom.sh
dnl #                             test/mpi/f08-mpi/ibcast.sh:test/mpi/testscripts/ibcast.sh
dnl #                             test/mpi/f08-mpi/ibcast_intercom.sh:test/mpi/testscripts/ibcast_intercom.sh
dnl #                             test/mpi/f08-mpi/gather.sh:test/mpi/testscripts/gather.sh
dnl #                             test/mpi/f08-mpi/gather_intercom.sh:test/mpi/testscripts/gather_intercom.sh
dnl #                             test/mpi/f08-mpi/gather_inplace.sh:test/mpi/testscripts/gather_inplace.sh
dnl #                             test/mpi/f08-mpi/igather.sh:test/mpi/testscripts/igather.sh
dnl #                             test/mpi/f08-mpi/igather_intercom.sh:test/mpi/testscripts/igather_intercom.sh
dnl #                             test/mpi/f08-mpi/igather_inplace.sh:test/mpi/testscripts/igather_inplace.sh
dnl #                             test/mpi/f08-mpi/gatherv.sh:test/mpi/testscripts/gatherv.sh
dnl #                             test/mpi/f08-mpi/gatherv_intercom.sh:test/mpi/testscripts/gatherv_intercom.sh
dnl #                             test/mpi/f08-mpi/gatherv_inplace.sh:test/mpi/testscripts/gatherv_inplace.sh
dnl #                             test/mpi/f08-mpi/igatherv.sh:test/mpi/testscripts/igatherv.sh
dnl #                             test/mpi/f08-mpi/igatherv_intercom.sh:test/mpi/testscripts/igatherv_intercom.sh
dnl #                             test/mpi/f08-mpi/igatherv_inplace.sh:test/mpi/testscripts/igatherv_inplace.sh
dnl #                             test/mpi/f08-mpi/scatter.sh:test/mpi/testscripts/scatter.sh
dnl #                             test/mpi/f08-mpi/scatter_intercom.sh:test/mpi/testscripts/scatter_intercom.sh
dnl #                             test/mpi/f08-mpi/scatter_inplace.sh:test/mpi/testscripts/scatter_inplace.sh
dnl #                             test/mpi/f08-mpi/iscatter.sh:test/mpi/testscripts/iscatter.sh
dnl #                             test/mpi/f08-mpi/iscatter_intercom.sh:test/mpi/testscripts/iscatter_intercom.sh
dnl #                             test/mpi/f08-mpi/iscatter_inplace.sh:test/mpi/testscripts/iscatter_inplace.sh
dnl #                             test/mpi/f08-mpi/scatterv.sh:test/mpi/testscripts/scatterv.sh
dnl #                             test/mpi/f08-mpi/scatterv_intercom.sh:test/mpi/testscripts/scatterv_intercom.sh
dnl #                             test/mpi/f08-mpi/scatterv_inplace.sh:test/mpi/testscripts/scatterv_inplace.sh
dnl #                             test/mpi/f08-mpi/iscatterv.sh:test/mpi/testscripts/iscatterv.sh
dnl #                             test/mpi/f08-mpi/iscatterv_intercom.sh:test/mpi/testscripts/iscatterv_intercom.sh
dnl #                             test/mpi/f08-mpi/iscatterv_inplace.sh:test/mpi/testscripts/iscatterv_inplace.sh
dnl #                             test/mpi/f08-mpi/allgather.sh:test/mpi/testscripts/allgather.sh
dnl #                             test/mpi/f08-mpi/allgather_intercom.sh:test/mpi/testscripts/allgather_intercom.sh
dnl #                             test/mpi/f08-mpi/allgather_inplace.sh:test/mpi/testscripts/allgather_inplace.sh
dnl #                             test/mpi/f08-mpi/iallgather.sh:test/mpi/testscripts/iallgather.sh
dnl #                             test/mpi/f08-mpi/iallgather_intercom.sh:test/mpi/testscripts/iallgather_intercom.sh
dnl #                             test/mpi/f08-mpi/iallgather_inplace.sh:test/mpi/testscripts/iallgather_inplace.sh
dnl #                             test/mpi/f08-mpi/allgatherv.sh:test/mpi/testscripts/allgatherv.sh
dnl #                             test/mpi/f08-mpi/allgatherv_intercom.sh:test/mpi/testscripts/allgatherv_intercom.sh
dnl #                             test/mpi/f08-mpi/allgatherv_inplace.sh:test/mpi/testscripts/allgatherv_inplace.sh
dnl #                             test/mpi/f08-mpi/iallgatherv.sh:test/mpi/testscripts/iallgatherv.sh
dnl #                             test/mpi/f08-mpi/iallgatherv_intercom.sh:test/mpi/testscripts/iallgatherv_intercom.sh
dnl #                             test/mpi/f08-mpi/iallgatherv_inplace.sh:test/mpi/testscripts/iallgatherv_inplace.sh
dnl #                             test/mpi/f08-mpi/alltoall.sh:test/mpi/testscripts/alltoall.sh
dnl #                             test/mpi/f08-mpi/alltoall_intercom.sh:test/mpi/testscripts/alltoall_intercom.sh
dnl #                             test/mpi/f08-mpi/alltoall_inplace.sh:test/mpi/testscripts/alltoall_inplace.sh
dnl #                             test/mpi/f08-mpi/ialltoall.sh:test/mpi/testscripts/ialltoall.sh
dnl #                             test/mpi/f08-mpi/ialltoall_intercom.sh:test/mpi/testscripts/ialltoall_intercom.sh
dnl #                             test/mpi/f08-mpi/ialltoall_inplace.sh:test/mpi/testscripts/ialltoall_inplace.sh
dnl #                             test/mpi/f08-mpi/alltoallv.sh:test/mpi/testscripts/alltoallv.sh
dnl #                             test/mpi/f08-mpi/alltoallv_intercom.sh:test/mpi/testscripts/alltoallv_intercom.sh
dnl #                             test/mpi/f08-mpi/alltoallv_inplace.sh:test/mpi/testscripts/alltoallv_inplace.sh
dnl #                             test/mpi/f08-mpi/ialltoallv.sh:test/mpi/testscripts/ialltoallv.sh
dnl #                             test/mpi/f08-mpi/ialltoallv_intercom.sh:test/mpi/testscripts/ialltoallv_intercom.sh
dnl #                             test/mpi/f08-mpi/ialltoallv_inplace.sh:test/mpi/testscripts/ialltoallv_inplace.sh
dnl #                             test/mpi/f08-mpi/alltoallw.sh:test/mpi/testscripts/alltoallw.sh
dnl #                             test/mpi/f08-mpi/alltoallw_intercom.sh:test/mpi/testscripts/alltoallw_intercom.sh
dnl #                             test/mpi/f08-mpi/alltoallw_inplace.sh:test/mpi/testscripts/alltoallw_inplace.sh
dnl #                             test/mpi/f08-mpi/ialltoallw.sh:test/mpi/testscripts/ialltoallw.sh
dnl #                             test/mpi/f08-mpi/ialltoallw_intercom.sh:test/mpi/testscripts/ialltoallw_intercom.sh
dnl #                             test/mpi/f08-mpi/ialltoallw_inplace.sh:test/mpi/testscripts/ialltoallw_inplace.sh
dnl #                             test/mpi/f08-mpi/reduce.sh:test/mpi/testscripts/reduce.sh
dnl #                             test/mpi/f08-mpi/reduce_intercom.sh:test/mpi/testscripts/reduce_intercom.sh
dnl #                             test/mpi/f08-mpi/reduce_inplace.sh:test/mpi/testscripts/reduce_inplace.sh
dnl #                             test/mpi/f08-mpi/ireduce.sh:test/mpi/testscripts/ireduce.sh
dnl #                             test/mpi/f08-mpi/ireduce_intercom.sh:test/mpi/testscripts/ireduce_intercom.sh
dnl #                             test/mpi/f08-mpi/ireduce_inplace.sh:test/mpi/testscripts/ireduce_inplace.sh
dnl #                             test/mpi/f08-mpi/allreduce.sh:test/mpi/testscripts/allreduce.sh
dnl #                             test/mpi/f08-mpi/allreduce_intercom.sh:test/mpi/testscripts/allreduce_intercom.sh
dnl #                             test/mpi/f08-mpi/allreduce_inplace.sh:test/mpi/testscripts/allreduce_inplace.sh
dnl #                             test/mpi/f08-mpi/iallreduce.sh:test/mpi/testscripts/iallreduce.sh
dnl #                             test/mpi/f08-mpi/iallreduce_intercom.sh:test/mpi/testscripts/iallreduce_intercom.sh
dnl #                             test/mpi/f08-mpi/iallreduce_inplace.sh:test/mpi/testscripts/iallreduce_inplace.sh
dnl #                             test/mpi/f08-mpi/reduce_scatter_block.sh:test/mpi/testscripts/reduce_scatter_block.sh
dnl #                             test/mpi/f08-mpi/reduce_scatter_block_intercom.sh:test/mpi/testscripts/reduce_scatter_block_intercom.sh
dnl #                             test/mpi/f08-mpi/reduce_scatter_block_inplace.sh:test/mpi/testscripts/reduce_scatter_block_inplace.sh
dnl #                             test/mpi/f08-mpi/ireduce_scatter_block.sh:test/mpi/testscripts/ireduce_scatter_block.sh
dnl #                             test/mpi/f08-mpi/ireduce_scatter_block_intercom.sh:test/mpi/testscripts/ireduce_scatter_block_intercom.sh
dnl #                             test/mpi/f08-mpi/ireduce_scatter_block_inplace.sh:test/mpi/testscripts/ireduce_scatter_block_inplace.sh
dnl #                             test/mpi/f08-mpi/reduce_scatter.sh:test/mpi/testscripts/reduce_scatter.sh
dnl #                             test/mpi/f08-mpi/reduce_scatter_intercom.sh:test/mpi/testscripts/reduce_scatter_intercom.sh
dnl #                             test/mpi/f08-mpi/reduce_scatter_inplace.sh:test/mpi/testscripts/reduce_scatter_inplace.sh
dnl #                             test/mpi/f08-mpi/ireduce_scatter.sh:test/mpi/testscripts/ireduce_scatter.sh
dnl #                             test/mpi/f08-mpi/ireduce_scatter_intercom.sh:test/mpi/testscripts/ireduce_scatter_intercom.sh
dnl #                             test/mpi/f08-mpi/ireduce_scatter_inplace.sh:test/mpi/testscripts/ireduce_scatter_inplace.sh
dnl #                             test/mpi/f08-mpi/scan.sh:test/mpi/testscripts/scan.sh
dnl #                             test/mpi/f08-mpi/scan_inplace.sh:test/mpi/testscripts/scan_inplace.sh
dnl #                             test/mpi/f08-mpi/iscan.sh:test/mpi/testscripts/iscan.sh
dnl #                             test/mpi/f08-mpi/iscan_inplace.sh:test/mpi/testscripts/iscan_inplace.sh
dnl #                             test/mpi/f08-mpi/exscan.sh:test/mpi/testscripts/exscan.sh
dnl #                             test/mpi/f08-mpi/exscan_inplace.sh:test/mpi/testscripts/exscan_inplace.sh
dnl #                             test/mpi/f08-mpi/iexscan.sh:test/mpi/testscripts/iexscan.sh
dnl #                             test/mpi/f08-mpi/iexscan_inplace.sh:test/mpi/testscripts/iexscan_inplace.sh
dnl #                             test/mpi/f08-mpi/put.sh:test/mpi/testscripts/put.sh
dnl #                             test/mpi/f08-mpi/rput.sh:test/mpi/testscripts/rput.sh
dnl #                             test/mpi/f08-mpi/get.sh:test/mpi/testscripts/get.sh
dnl #                             test/mpi/f08-mpi/rget.sh:test/mpi/testscripts/rget.sh
dnl #                             test/mpi/f08-mpi/accumulate.sh:test/mpi/testscripts/accumulate.sh
dnl #                             test/mpi/f08-mpi/raccumulate.sh:test/mpi/testscripts/raccumulate.sh
dnl #                             test/mpi/f08-mpi/get_accumulate.sh:test/mpi/testscripts/get_accumulate.sh
dnl #                             test/mpi/f08-mpi/rget_accumulate.sh:test/mpi/testscripts/rget_accumulate.sh
dnl #                             test/mpi/f08-mpi/fetch_and_op.sh:test/mpi/testscripts/fetch_and_op.sh
dnl #                             test/mpi/f08-mpi/compare_and_swap.sh:test/mpi/testscripts/compare_and_swap.sh
dnl #                             test/mpi/f08-mpi/test.sh:test/mpi/testscripts/test.sh
dnl #                             test/mpi/f08-mpi/testany.sh:test/mpi/testscripts/testany.sh
dnl #                             test/mpi/f08-mpi/testsome.sh:test/mpi/testscripts/testsome.sh
dnl #                             test/mpi/f08-mpi/testall.sh:test/mpi/testscripts/testall.sh
dnl #                             test/mpi/f08-mpi/wait.sh:test/mpi/testscripts/wait.sh
dnl #                             test/mpi/f08-mpi/waitany.sh:test/mpi/testscripts/waitany.sh
dnl #                             test/mpi/f08-mpi/waitsome.sh:test/mpi/testscripts/waitsome.sh
dnl #                             test/mpi/f08-mpi/waitall.sh:test/mpi/testscripts/waitall.sh
dnl #                            )
dnl #            ])
               
AC_OUTPUT

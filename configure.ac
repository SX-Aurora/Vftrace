AC_INIT([vftrace], [1.0], [foo@foo.org])
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
AC_ARG_ENABLE([papi],
  [AS_HELP_STRING([--enable-papi], [enable PAPI [[no]]])],
  [], [enable_papi="no"])

AC_ARG_ENABLE([veperf],
  [AS_HELP_STRING([--enable-veperf], [enable VEPERF for SX Aurora [[no]]])],
  [], [enable_veperf="no"])

AC_ARG_ENABLE([mpi_profiling],
  [AS_HELP_STRING([--enable-mpi-profiling], [enable MPI profiling [[no]]])],
  [], [enable_mpi_profiling="no"])

AC_ARG_ENABLE([vmap_offset],
  [AS_HELP_STRING([--enable-vmap-offset], [enable VMAP offset [[no]]])],
  [], [enable_vmap_offset="no"])

AC_ARG_ENABLE([vmap_offset_check],
  [AS_HELP_STRING([--enable-vmap-offset-check], [enable VMAP offset check [[yes]]])],
  [], [enable_vmap_offset_check="yes"])

AC_ARG_WITH([papi],
  [AS_HELP_STRING([--with-papi=dir], [assume given directory for PAPI])])

AC_ARG_WITH([veperf],
  [AS_HELP_STRING([--with-veperf=dir], [assume given directory for Veperf])])

AM_CONDITIONAL([PAPI_AVAIL], [test "$enable_papi" = "yes"])
PAPI_HOME=$with_papi
AC_SUBST([PAPI_HOME])

AM_CONDITIONAL([VEPERF_AVAIL], [test "$enable_veperf" = "yes"])
VEPERF_HOME=$with_veperf
AC_SUBST([VEPERF_HOME])

AM_CONDITIONAL([VMAP_OFFSET], [test "$enable_vmap_offset" = "yes"])
AM_CONDITIONAL([MPI_PROFILING], [test "$enable_mpi_profiling" = "yes"])
AM_CONDITIONAL([VMAP_OFFSET_CHECK], [test "$enable_vmap_offset_check" = "yes"])
################################

if test "$enable_vmap_offset_check" = yes; then
  AC_MSG_CHECKING([if virtual map offset has to be used])
  doit='src/aux/config_test.sh'
  if AC_TRY_EVAL(doit); then
     use_vmap_offset=yes
  else
     use_vmap_offset=no
  fi 
  AC_MSG_RESULT([$use_vmap_offset])
fi

if test "$enable_mpi_profiling" = yes; then
  AC_MSG_CHECKING([if Intel MPI is used])
  check_if_intel_mpi='mpirun --version | grep Intel'
  if AC_TRY_EVAL(check_if_intel_mpi); then
    uses_intel_mpi=yes
  else
    uses_intel_mpi=no
  fi
  AC_MSG_RESULT([$uses_intel_mpi])
  AC_MSG_CHECKING([if NEC MPI is used])
  check_if_nec_mpi='mpirun --version | grep NEC'
  if AC_TRY_EVAL(check_if_nec_mpi); then
    uses_nec_mpi=yes
  else
    uses_nec_mpi=no
  fi
fi 

AC_MSG_CHECKING([if Intel compiler is used])
check_if_intel_compiler='$CC --version | grep Intel'
if AC_TRY_EVAL(check_if_intel_compiler); then
  uses_intel_compiler=yes
else
  uses_intel_compiler=no
fi 
AC_MSG_RESULT([$uses_intel_compiler])

AM_CONDITIONAL([USES_INTEL_MPI], [test "$uses_intel_mpi" = "yes"])
AM_CONDITIONAL([USES_NEC_MPI], [test "$uses_nec_mpi" = "yes"])
AM_CONDITIONAL([USES_INTEL_COMPILER], [test "$uses_intel_compiler" = "yes"])

# Check if Fortran-MPI supports TS29113
# Fortran 90:
AC_LANG(Fortran)
AM_COND_IF(
   [MPI_PROFILING],
   [AC_MSG_CHECKING([if MPI-F90 supports TS29113])
    AC_RUN_IFELSE(
      [AC_LANG_SOURCE([[
PROGRAM test
   USE mpi
   IF (MPI_SUBARRAYS_SUPPORTED) STOP 1
   STOP 0
END PROGRAM test]])],
      [uses_mpi_f90_ts29113=no],
      [uses_mpi_f90_ts29113=yes])]
)
AM_CONDITIONAL([SUPPORTS_MPIF90_TS], [test "$supports_mpi_f90_ts29113" = "yes"])

# Fortran 2008:
AM_COND_IF(
   [MPI_PROFILING],
   [AC_MSG_CHECKING([if MPI-F08 supports TS29113])
    AC_RUN_IFELSE(
      [AC_LANG_SOURCE([[
PROGRAM test
   USE mpi_f08
   IF (MPI_SUBARRAYS_SUPPORTED) STOP 1
   STOP 0
END PROGRAM test]])],
      [uses_mpi_f08_ts29113=no],
      [uses_mpi_f08_ts29113=yes])]
)
AM_CONDITIONAL([SUPPORTS_MPIF08_TS], [test "$supports_mpi_f08_ts29113" = "yes"])



# WO_PROG_PAPI()
# WO_PROG_COMPONENTS()
# Enable libtool
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
AC_CONFIG_FILES([test/user_fkt/Makefile])
AC_CONFIG_FILES([test/mpi/Makefile])
AC_CONFIG_FILES([test/mpi/c-mpi/Makefile])
AC_CONFIG_FILES([test/mpi/f-mpi/Makefile])
AC_CONFIG_FILES([test/mpi/f08-mpi/Makefile])

# Test files and folders
AC_CONFIG_LINKS(test/units/ref_input/vftr_scenario_test_1.json:test/units/ref_input/vftr_scenario_test_1.json
                test/units/ref_input/vftr_scenario_test_2.json:test/units/ref_input/vftr_scenario_test_2.json
                test/units/ref_input/vftr_scenario_test_3.json:test/units/ref_input/vftr_scenario_test_3.json
                test/units/ref_input/vftr_symbols_test_1.x:test/units/ref_input/vftr_symbols_test_1.x
               )
AC_CONFIG_LINKS(test/units/ref_output/vftr_environment_test_1.out:test/units/ref_output/vftr_environment_test_1.out
                test/units/ref_output/vftr_filewrite_test_1.out:test/units/ref_output/vftr_filewrite_test_1.out
                test/units/ref_output/vftr_scenario_test_1.out:test/units/ref_output/vftr_scenario_test_1.out
                test/units/ref_output/vftr_scenario_test_2.out:test/units/ref_output/vftr_scenario_test_2.out
                test/units/ref_output/vftr_setup_test_1.out:test/units/ref_output/vftr_setup_test_1.out
                test/units/ref_output/vftr_setup_test_2.out:test/units/ref_output/vftr_setup_test_2.out
                test/units/ref_output/vftr_symbols_test_1.out:test/units/ref_output/vftr_symbols_test_1.out
               )

AM_COND_IF([MPI_PROFILING],[units_refdir=mpi], [units_refdir=serial])

AC_CONFIG_LINKS(test/units/ref_output/vftr_functions_test_1.out:test/units/ref_output/$units_refdir/vftr_functions_test_1.out
                test/units/ref_output/vftr_functions_test_2.out:test/units/ref_output/$units_refdir/vftr_functions_test_2.out
                test/units/ref_output/vftr_functions_test_3.out:test/units/ref_output/$units_refdir/vftr_functions_test_3.out
                test/units/ref_output/vftr_functions_test_4.out:test/units/ref_output/$units_refdir/vftr_functions_test_4.out
                test/units/ref_output/vftr_filewrite_test_2.out:test/units/ref_output/$units_refdir/vftr_filewrite_test_2.out
                test/units/ref_output/vftr_stacks_test_1.out:test/units/ref_output/$units_refdir/vftr_stacks_test_1.out)

AC_CONFIG_LINKS(test/units/this_passes.sh:test/units/this_passes.sh
                test/units/this_fails.sh:test/units/this_fails.sh
                test/units/vftr_symbols_test_1.sh:test/units/vftr_symbols_test_1.sh
                test/units/vftr_environment_test_1.sh:test/units/vftr_environment_test_1.sh
                test/units/vftr_setup_test_1.sh:test/units/vftr_setup_test_1.sh
                test/units/vftr_setup_test_2.sh:test/units/vftr_setup_test_2.sh
                test/units/vftr_filewrite_test_1.sh:test/units/vftr_filewrite_test_1.sh
                test/units/vftr_filewrite_test_2.sh:test/units/vftr_filewrite_test_2.sh
                test/units/vftr_scenario_test_1.sh:test/units/vftr_scenario_test_1.sh
                test/units/vftr_scenario_test_2.sh:test/units/vftr_scenario_test_2.sh
                test/units/vftr_scenario_test_3.sh:test/units/vftr_scenario_test_3.sh
                test/units/vftr_functions_test_1.sh:test/units/vftr_functions_test_1.sh
                test/units/vftr_functions_test_2.sh:test/units/vftr_functions_test_2.sh
                test/units/vftr_functions_test_3.sh:test/units/vftr_functions_test_3.sh
                test/units/vftr_functions_test_4.sh:test/units/vftr_functions_test_4.sh
                test/units/vftr_stacks_test_1.sh:test/units/vftr_stacks_test_1.sh
               )

# User funktion tests
AC_CONFIG_LINKS(test/user_fkt/cregions1.sh:test/user_fkt/cregions1.sh
                test/user_fkt/cregions2.sh:test/user_fkt/cregions2.sh
                test/user_fkt/cregions3.sh:test/user_fkt/cregions3.sh
                test/user_fkt/cregions4.sh:test/user_fkt/cregions4.sh
                test/user_fkt/fregions1.sh:test/user_fkt/fregions1.sh
                test/user_fkt/fregions2.sh:test/user_fkt/fregions2.sh
                test/user_fkt/fregions3.sh:test/user_fkt/fregions3.sh
                test/user_fkt/fregions4.sh:test/user_fkt/fregions4.sh
               )

# C-MPI testscripts
AM_COND_IF([MPI_PROFILING],
           [AC_CONFIG_LINKS(test/mpi/c-mpi/init_finalize-1.sh:test/mpi/testscripts/init_finalize-1.sh
                            test/mpi/c-mpi/init_finalize-2.sh:test/mpi/testscripts/init_finalize-2.sh
                            test/mpi/c-mpi/init_finalize-3.sh:test/mpi/testscripts/init_finalize-3.sh
                            test/mpi/c-mpi/init_finalize-4.sh:test/mpi/testscripts/init_finalize-4.sh
                            test/mpi/c-mpi/ctypes.sh:test/mpi/testscripts/ctypes.sh
                            test/mpi/c-mpi/derived_ctypes.sh:test/mpi/testscripts/derived_ctypes.sh
                            test/mpi/c-mpi/send_recv.sh:test/mpi/testscripts/send_recv.sh
                            test/mpi/c-mpi/sendrecv.sh:test/mpi/testscripts/sendrecv.sh
                            test/mpi/c-mpi/sendrecv_replace.sh:test/mpi/testscripts/sendrecv_replace.sh
                            test/mpi/c-mpi/isend_recv.sh:test/mpi/testscripts/isend_recv.sh
                            test/mpi/c-mpi/send_irecv.sh:test/mpi/testscripts/send_irecv.sh
                            test/mpi/c-mpi/bcast.sh:test/mpi/testscripts/bcast.sh
                            test/mpi/c-mpi/bcast_intercom.sh:test/mpi/testscripts/bcast_intercom.sh
                            test/mpi/c-mpi/ibcast.sh:test/mpi/testscripts/ibcast.sh
                            test/mpi/c-mpi/ibcast_intercom.sh:test/mpi/testscripts/ibcast_intercom.sh
                            test/mpi/c-mpi/gather.sh:test/mpi/testscripts/gather.sh
                            test/mpi/c-mpi/gather_intercom.sh:test/mpi/testscripts/gather_intercom.sh
                            test/mpi/c-mpi/igather.sh:test/mpi/testscripts/igather.sh
                            test/mpi/c-mpi/igather_intercom.sh:test/mpi/testscripts/igather_intercom.sh
                            test/mpi/c-mpi/gatherv.sh:test/mpi/testscripts/gatherv.sh
                            test/mpi/c-mpi/gatherv_intercom.sh:test/mpi/testscripts/gatherv_intercom.sh
                            test/mpi/c-mpi/igatherv.sh:test/mpi/testscripts/igatherv.sh
                            test/mpi/c-mpi/igatherv_intercom.sh:test/mpi/testscripts/igatherv_intercom.sh
                            test/mpi/c-mpi/scatter.sh:test/mpi/testscripts/scatter.sh
                            test/mpi/c-mpi/scatter_intercom.sh:test/mpi/testscripts/scatter_intercom.sh
                            test/mpi/c-mpi/iscatter.sh:test/mpi/testscripts/iscatter.sh
                            test/mpi/c-mpi/iscatter_intercom.sh:test/mpi/testscripts/iscatter_intercom.sh
                            test/mpi/c-mpi/scatterv.sh:test/mpi/testscripts/scatterv.sh
                            test/mpi/c-mpi/scatterv_intercom.sh:test/mpi/testscripts/scatterv_intercom.sh
                            test/mpi/c-mpi/iscatterv.sh:test/mpi/testscripts/iscatterv.sh
                            test/mpi/c-mpi/iscatterv_intercom.sh:test/mpi/testscripts/iscatterv_intercom.sh
                            test/mpi/c-mpi/allgather.sh:test/mpi/testscripts/allgather.sh
                            test/mpi/c-mpi/allgather_intercom.sh:test/mpi/testscripts/allgather_intercom.sh
                            test/mpi/c-mpi/iallgather.sh:test/mpi/testscripts/iallgather.sh
                            test/mpi/c-mpi/iallgather_intercom.sh:test/mpi/testscripts/iallgather_intercom.sh
                            test/mpi/c-mpi/allgatherv.sh:test/mpi/testscripts/allgatherv.sh
                            test/mpi/c-mpi/allgatherv_intercom.sh:test/mpi/testscripts/allgatherv_intercom.sh
                            test/mpi/c-mpi/iallgatherv.sh:test/mpi/testscripts/iallgatherv.sh
                            test/mpi/c-mpi/iallgatherv_intercom.sh:test/mpi/testscripts/iallgatherv_intercom.sh
                            test/mpi/c-mpi/alltoall.sh:test/mpi/testscripts/alltoall.sh
                            test/mpi/c-mpi/alltoall_intercom.sh:test/mpi/testscripts/alltoall_intercom.sh
                            test/mpi/c-mpi/ialltoall.sh:test/mpi/testscripts/ialltoall.sh
                            test/mpi/c-mpi/ialltoall_intercom.sh:test/mpi/testscripts/ialltoall_intercom.sh
                            test/mpi/c-mpi/alltoallv.sh:test/mpi/testscripts/alltoallv.sh
                            test/mpi/c-mpi/alltoallv_intercom.sh:test/mpi/testscripts/alltoallv_intercom.sh
                            test/mpi/c-mpi/ialltoallv.sh:test/mpi/testscripts/ialltoallv.sh
                            test/mpi/c-mpi/ialltoallv_intercom.sh:test/mpi/testscripts/ialltoallv_intercom.sh
                            test/mpi/c-mpi/alltoallw.sh:test/mpi/testscripts/alltoallw.sh
                            test/mpi/c-mpi/alltoallw_intercom.sh:test/mpi/testscripts/alltoallw_intercom.sh
                            test/mpi/c-mpi/ialltoallw.sh:test/mpi/testscripts/ialltoallw.sh
                            test/mpi/c-mpi/ialltoallw_intercom.sh:test/mpi/testscripts/ialltoallw_intercom.sh
                            test/mpi/c-mpi/reduce.sh:test/mpi/testscripts/reduce.sh
                            test/mpi/c-mpi/reduce_intercom.sh:test/mpi/testscripts/reduce_intercom.sh
                            test/mpi/c-mpi/allreduce.sh:test/mpi/testscripts/allreduce.sh
                            test/mpi/c-mpi/allreduce_intercom.sh:test/mpi/testscripts/allreduce_intercom.sh
                            test/mpi/c-mpi/put.sh:test/mpi/testscripts/put.sh
                            test/mpi/c-mpi/get.sh:test/mpi/testscripts/get.sh
                            test/mpi/c-mpi/test.sh:test/mpi/testscripts/test.sh
                            test/mpi/c-mpi/testany.sh:test/mpi/testscripts/testany.sh
                            test/mpi/c-mpi/testsome.sh:test/mpi/testscripts/testsome.sh
                            test/mpi/c-mpi/testall.sh:test/mpi/testscripts/testall.sh
                            test/mpi/c-mpi/wait.sh:test/mpi/testscripts/wait.sh
                            test/mpi/c-mpi/waitany.sh:test/mpi/testscripts/waitany.sh
                            test/mpi/c-mpi/waitsome.sh:test/mpi/testscripts/waitsome.sh
                            test/mpi/c-mpi/waitall.sh:test/mpi/testscripts/waitall.sh
                           )
           ])
                
# F-MPI testscripts
AM_COND_IF([MPI_PROFILING],
           [AC_CONFIG_LINKS(test/mpi/f-mpi/init_finalize-1.sh:test/mpi/testscripts/init_finalize-1.sh
                            test/mpi/f-mpi/init_finalize-2.sh:test/mpi/testscripts/init_finalize-2.sh
                            test/mpi/f-mpi/init_finalize-3.sh:test/mpi/testscripts/init_finalize-3.sh
                            test/mpi/f-mpi/init_finalize-4.sh:test/mpi/testscripts/init_finalize-4.sh
                            test/mpi/f-mpi/ftypes.sh:test/mpi/testscripts/ftypes.sh
                            test/mpi/f-mpi/derived_ftypes.sh:test/mpi/testscripts/derived_ftypes.sh
                            test/mpi/f-mpi/send_recv.sh:test/mpi/testscripts/send_recv.sh
                            test/mpi/f-mpi/sendrecv.sh:test/mpi/testscripts/sendrecv.sh
                            test/mpi/f-mpi/sendrecv_replace.sh:test/mpi/testscripts/sendrecv_replace.sh
                            test/mpi/f-mpi/isend_recv.sh:test/mpi/testscripts/isend_recv.sh
                            test/mpi/f-mpi/send_irecv.sh:test/mpi/testscripts/send_irecv.sh
                            test/mpi/f-mpi/bcast.sh:test/mpi/testscripts/bcast.sh
                            test/mpi/f-mpi/bcast_intercom.sh:test/mpi/testscripts/bcast_intercom.sh
                            test/mpi/f-mpi/ibcast.sh:test/mpi/testscripts/ibcast.sh
                            test/mpi/f-mpi/ibcast_intercom.sh:test/mpi/testscripts/ibcast_intercom.sh
                            test/mpi/f-mpi/gather.sh:test/mpi/testscripts/gather.sh
                            test/mpi/f-mpi/gather_intercom.sh:test/mpi/testscripts/gather_intercom.sh
                            test/mpi/f-mpi/igather.sh:test/mpi/testscripts/igather.sh
                            test/mpi/f-mpi/igather_intercom.sh:test/mpi/testscripts/igather_intercom.sh
                            test/mpi/f-mpi/gatherv.sh:test/mpi/testscripts/gatherv.sh
                            test/mpi/f-mpi/gatherv_intercom.sh:test/mpi/testscripts/gatherv_intercom.sh
                            test/mpi/f-mpi/igatherv.sh:test/mpi/testscripts/igatherv.sh
                            test/mpi/f-mpi/igatherv_intercom.sh:test/mpi/testscripts/igatherv_intercom.sh
                            test/mpi/f-mpi/scatter.sh:test/mpi/testscripts/scatter.sh
                            test/mpi/f-mpi/scatter_intercom.sh:test/mpi/testscripts/scatter_intercom.sh
                            test/mpi/f-mpi/iscatter.sh:test/mpi/testscripts/iscatter.sh
                            test/mpi/f-mpi/iscatter_intercom.sh:test/mpi/testscripts/iscatter_intercom.sh
                            test/mpi/f-mpi/scatterv.sh:test/mpi/testscripts/scatterv.sh
                            test/mpi/f-mpi/scatterv_intercom.sh:test/mpi/testscripts/scatterv_intercom.sh
                            test/mpi/f-mpi/iscatterv.sh:test/mpi/testscripts/iscatterv.sh
                            test/mpi/f-mpi/iscatterv_intercom.sh:test/mpi/testscripts/iscatterv_intercom.sh
                            test/mpi/f-mpi/allgather.sh:test/mpi/testscripts/allgather.sh
                            test/mpi/f-mpi/allgather_intercom.sh:test/mpi/testscripts/allgather_intercom.sh
                            test/mpi/f-mpi/iallgather.sh:test/mpi/testscripts/iallgather.sh
                            test/mpi/f-mpi/iallgather_intercom.sh:test/mpi/testscripts/iallgather_intercom.sh
                            test/mpi/f-mpi/allgatherv.sh:test/mpi/testscripts/allgatherv.sh
                            test/mpi/f-mpi/allgatherv_intercom.sh:test/mpi/testscripts/allgatherv_intercom.sh
                            test/mpi/f-mpi/iallgatherv.sh:test/mpi/testscripts/iallgatherv.sh
                            test/mpi/f-mpi/iallgatherv_intercom.sh:test/mpi/testscripts/iallgatherv_intercom.sh
                            test/mpi/f-mpi/alltoall.sh:test/mpi/testscripts/alltoall.sh
                            test/mpi/f-mpi/alltoall_intercom.sh:test/mpi/testscripts/alltoall_intercom.sh
                            test/mpi/f-mpi/ialltoall.sh:test/mpi/testscripts/ialltoall.sh
                            test/mpi/f-mpi/ialltoall_intercom.sh:test/mpi/testscripts/ialltoall_intercom.sh
                            test/mpi/f-mpi/alltoallv.sh:test/mpi/testscripts/alltoallv.sh
                            test/mpi/f-mpi/alltoallv_intercom.sh:test/mpi/testscripts/alltoallv_intercom.sh
                            test/mpi/f-mpi/ialltoallv.sh:test/mpi/testscripts/ialltoallv.sh
                            test/mpi/f-mpi/ialltoallv_intercom.sh:test/mpi/testscripts/ialltoallv_intercom.sh
                            test/mpi/f-mpi/alltoallw.sh:test/mpi/testscripts/alltoallw.sh
                            test/mpi/f-mpi/alltoallw_intercom.sh:test/mpi/testscripts/alltoallw_intercom.sh
                            test/mpi/f-mpi/ialltoallw.sh:test/mpi/testscripts/ialltoallw.sh
                            test/mpi/f-mpi/ialltoallw_intercom.sh:test/mpi/testscripts/ialltoallw_intercom.sh
                            test/mpi/f-mpi/reduce.sh:test/mpi/testscripts/reduce.sh
                            test/mpi/f-mpi/reduce_intercom.sh:test/mpi/testscripts/reduce_intercom.sh
                            test/mpi/f-mpi/allreduce.sh:test/mpi/testscripts/allreduce.sh
                            test/mpi/f-mpi/allreduce_intercom.sh:test/mpi/testscripts/allreduce_intercom.sh
                            test/mpi/f-mpi/put.sh:test/mpi/testscripts/put.sh
                            test/mpi/f-mpi/get.sh:test/mpi/testscripts/get.sh
                            test/mpi/f-mpi/test.sh:test/mpi/testscripts/test.sh
                            test/mpi/f-mpi/testany.sh:test/mpi/testscripts/testany.sh
                            test/mpi/f-mpi/testsome.sh:test/mpi/testscripts/testsome.sh
                            test/mpi/f-mpi/testall.sh:test/mpi/testscripts/testall.sh
                            test/mpi/f-mpi/wait.sh:test/mpi/testscripts/wait.sh
                            test/mpi/f-mpi/waitany.sh:test/mpi/testscripts/waitany.sh
                            test/mpi/f-mpi/waitsome.sh:test/mpi/testscripts/waitsome.sh
                            test/mpi/f-mpi/waitall.sh:test/mpi/testscripts/waitall.sh
                           )
           ])
                               
# F08-MPI testscripts
AM_COND_IF([MPI_PROFILING],
           [AC_CONFIG_LINKS(test/mpi/f08-mpi/init_finalize-1.sh:test/mpi/testscripts/init_finalize-1.sh
                            test/mpi/f08-mpi/init_finalize-2.sh:test/mpi/testscripts/init_finalize-2.sh
                            test/mpi/f08-mpi/init_finalize-3.sh:test/mpi/testscripts/init_finalize-3.sh
                            test/mpi/f08-mpi/init_finalize-4.sh:test/mpi/testscripts/init_finalize-4.sh
                            test/mpi/f08-mpi/ftypes.sh:test/mpi/testscripts/ftypes.sh
                            test/mpi/f08-mpi/derived_ftypes.sh:test/mpi/testscripts/derived_ftypes.sh
                            test/mpi/f08-mpi/send_recv.sh:test/mpi/testscripts/send_recv.sh
                            test/mpi/f08-mpi/sendrecv.sh:test/mpi/testscripts/sendrecv.sh
                            test/mpi/f08-mpi/sendrecv_replace.sh:test/mpi/testscripts/sendrecv_replace.sh
                            test/mpi/f08-mpi/isend_recv.sh:test/mpi/testscripts/isend_recv.sh
                            test/mpi/f08-mpi/send_irecv.sh:test/mpi/testscripts/send_irecv.sh
                            test/mpi/f08-mpi/bcast.sh:test/mpi/testscripts/bcast.sh
                            test/mpi/f08-mpi/bcast_intercom.sh:test/mpi/testscripts/bcast_intercom.sh
                            test/mpi/f08-mpi/ibcast.sh:test/mpi/testscripts/ibcast.sh
                            test/mpi/f08-mpi/ibcast_intercom.sh:test/mpi/testscripts/ibcast_intercom.sh
                            test/mpi/f08-mpi/gather.sh:test/mpi/testscripts/gather.sh
                            test/mpi/f08-mpi/gather_intercom.sh:test/mpi/testscripts/gather_intercom.sh
                            test/mpi/f08-mpi/igather.sh:test/mpi/testscripts/igather.sh
                            test/mpi/f08-mpi/igather_intercom.sh:test/mpi/testscripts/igather_intercom.sh
                            test/mpi/f08-mpi/gatherv.sh:test/mpi/testscripts/gatherv.sh
                            test/mpi/f08-mpi/gatherv_intercom.sh:test/mpi/testscripts/gatherv_intercom.sh
                            test/mpi/f08-mpi/igatherv.sh:test/mpi/testscripts/igatherv.sh
                            test/mpi/f08-mpi/igatherv_intercom.sh:test/mpi/testscripts/igatherv_intercom.sh
                            test/mpi/f08-mpi/scatter.sh:test/mpi/testscripts/scatter.sh
                            test/mpi/f08-mpi/scatter_intercom.sh:test/mpi/testscripts/scatter_intercom.sh
                            test/mpi/f08-mpi/iscatter.sh:test/mpi/testscripts/iscatter.sh
                            test/mpi/f08-mpi/iscatter_intercom.sh:test/mpi/testscripts/iscatter_intercom.sh
                            test/mpi/f08-mpi/scatterv.sh:test/mpi/testscripts/scatterv.sh
                            test/mpi/f08-mpi/scatterv_intercom.sh:test/mpi/testscripts/scatterv_intercom.sh
                            test/mpi/f08-mpi/iscatterv.sh:test/mpi/testscripts/iscatterv.sh
                            test/mpi/f08-mpi/iscatterv_intercom.sh:test/mpi/testscripts/iscatterv_intercom.sh
                            test/mpi/f08-mpi/allgather.sh:test/mpi/testscripts/allgather.sh
                            test/mpi/f08-mpi/allgather_intercom.sh:test/mpi/testscripts/allgather_intercom.sh
                            test/mpi/f08-mpi/iallgather.sh:test/mpi/testscripts/iallgather.sh
                            test/mpi/f08-mpi/iallgather_intercom.sh:test/mpi/testscripts/iallgather_intercom.sh
                            test/mpi/f08-mpi/allgatherv.sh:test/mpi/testscripts/allgatherv.sh
                            test/mpi/f08-mpi/allgatherv_intercom.sh:test/mpi/testscripts/allgatherv_intercom.sh
                            test/mpi/f08-mpi/iallgatherv.sh:test/mpi/testscripts/iallgatherv.sh
                            test/mpi/f08-mpi/iallgatherv_intercom.sh:test/mpi/testscripts/iallgatherv_intercom.sh
                            test/mpi/f08-mpi/alltoall.sh:test/mpi/testscripts/alltoall.sh
                            test/mpi/f08-mpi/alltoall_intercom.sh:test/mpi/testscripts/alltoall_intercom.sh
                            test/mpi/f08-mpi/ialltoall.sh:test/mpi/testscripts/ialltoall.sh
                            test/mpi/f08-mpi/ialltoall_intercom.sh:test/mpi/testscripts/ialltoall_intercom.sh
                            test/mpi/f08-mpi/alltoallv.sh:test/mpi/testscripts/alltoallv.sh
                            test/mpi/f08-mpi/alltoallv_intercom.sh:test/mpi/testscripts/alltoallv_intercom.sh
                            test/mpi/f08-mpi/ialltoallv.sh:test/mpi/testscripts/ialltoallv.sh
                            test/mpi/f08-mpi/ialltoallv_intercom.sh:test/mpi/testscripts/ialltoallv_intercom.sh
                            test/mpi/f08-mpi/alltoallw.sh:test/mpi/testscripts/alltoallw.sh
                            test/mpi/f08-mpi/alltoallw_intercom.sh:test/mpi/testscripts/alltoallw_intercom.sh
                            test/mpi/f08-mpi/ialltoallw.sh:test/mpi/testscripts/ialltoallw.sh
                            test/mpi/f08-mpi/ialltoallw_intercom.sh:test/mpi/testscripts/ialltoallw_intercom.sh
                            test/mpi/f08-mpi/reduce.sh:test/mpi/testscripts/reduce.sh
                            test/mpi/f08-mpi/reduce_intercom.sh:test/mpi/testscripts/reduce_intercom.sh
                            test/mpi/f08-mpi/allreduce.sh:test/mpi/testscripts/allreduce.sh
                            test/mpi/f08-mpi/allreduce_intercom.sh:test/mpi/testscripts/allreduce_intercom.sh
                            test/mpi/f08-mpi/put.sh:test/mpi/testscripts/put.sh
                            test/mpi/f08-mpi/get.sh:test/mpi/testscripts/get.sh
                            test/mpi/f08-mpi/test.sh:test/mpi/testscripts/test.sh
                            test/mpi/f08-mpi/testany.sh:test/mpi/testscripts/testany.sh
                            test/mpi/f08-mpi/testsome.sh:test/mpi/testscripts/testsome.sh
                            test/mpi/f08-mpi/testall.sh:test/mpi/testscripts/testall.sh
                            test/mpi/f08-mpi/wait.sh:test/mpi/testscripts/wait.sh
                            test/mpi/f08-mpi/waitany.sh:test/mpi/testscripts/waitany.sh
                            test/mpi/f08-mpi/waitsome.sh:test/mpi/testscripts/waitsome.sh
                            test/mpi/f08-mpi/waitall.sh:test/mpi/testscripts/waitall.sh
                           )
           ])
               
AC_OUTPUT

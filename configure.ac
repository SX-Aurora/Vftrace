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
AC_CONFIG_FILES([test/c-mpi/Makefile])
#AC_CONFIG_FILES([test/f-mpi/Makefile])
#AC_CONFIG_FILES([test/f08-mpi/Makefile])

# Test files and folders
AC_CONFIG_LINKS(test/units/ref_input/vftr_scenario_test_1.json:test/units/ref_input/vftr_scenario_test_1.json
                test/units/ref_input/vftr_scenario_test_2.json:test/units/ref_input/vftr_scenario_test_2.json
                test/units/ref_input/vftr_scenario_test_3.json:test/units/ref_input/vftr_scenario_test_3.json
                test/units/ref_input/vftr_symbols_test_1.x:test/units/ref_input/vftr_symbols_test_1.x
               )
AC_CONFIG_LINKS(test/units/ref_output/vftr_environment_test_1.out:test/units/ref_output/vftr_environment_test_1.out
                test/units/ref_output/vftr_filewrite_test_1.out:test/units/ref_output/vftr_filewrite_test_1.out
                test/units/ref_output/vftr_filewrite_test_2.out:test/units/ref_output/vftr_filewrite_test_2.out
                test/units/ref_output/vftr_scenario_test_1.out:test/units/ref_output/vftr_scenario_test_1.out
                test/units/ref_output/vftr_scenario_test_2.out:test/units/ref_output/vftr_scenario_test_2.out
                test/units/ref_output/vftr_setup_test_1.out:test/units/ref_output/vftr_setup_test_1.out
                test/units/ref_output/vftr_setup_test_2.out:test/units/ref_output/vftr_setup_test_2.out
                test/units/ref_output/vftr_stacks_test_1.out:test/units/ref_output/vftr_stacks_test_1.out
                test/units/ref_output/vftr_symbols_test_1.out:test/units/ref_output/vftr_symbols_test_1.out
               )

AM_COND_IF([MPI_PROFILING],[units_refdir=mpi], [units_refdir=serial])

AC_CONFIG_LINKS(test/units/ref_output/vftr_functions_test_1.out:test/units/ref_output/$units_refdir/vftr_functions_test_1.out
                test/units/ref_output/vftr_functions_test_2.out:test/units/ref_output/$units_refdir/vftr_functions_test_2.out
                test/units/ref_output/vftr_functions_test_3.out:test/units/ref_output/$units_refdir/vftr_functions_test_3.out)

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
                test/units/vftr_stacks_test_1.sh:test/units/vftr_stacks_test_1.sh
               )
               
AC_OUTPUT

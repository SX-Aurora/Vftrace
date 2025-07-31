AC_INIT([Vftrace], [2.0.0], [felix.uhl@ruhr-uni-bochum.de.com,christian.weiss@emea.nec.com])
: ${CFLAGS="-O2"}
: ${FCFLAGS="-O2"}

# Will be generated
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])
AM_INIT_AUTOMAKE([-Wall -Werror foreign no-dependencies subdir-objects])
AC_PROG_CC
AC_PROG_FC
AC_PROG_CXX
AC_FC_SRCEXT(F90)
AC_ENABLE_SHARED
AX_ENABLE_SHARED
AM_PROG_AR
# Enable per-target flags for C
# According to the documentation, this should be obsolete
# but without, autoreconf explicitly asks for it ... ?
LT_INIT
AM_PROG_CC_C_O
# WO_PROG_COMPILERS()
################################
# Check if C-compiler supports c99 standard
AX_CHECK_C99_STD

AX_CHECK_CXX_STDFLAG

AX_CHECK_FINSTRUMENT_OPTION_C
AX_CHECK_FINSTRUMENT_OPTION_FORTRAN

# Check some sizeof values
AX_SIZEOF_TYPES

# check if the uintptr_t type is supported
AX_CHECK_UINTPTR

# Check if fortran is selected or not
AX_ENABLE_FORTRAN

# Check if on VE or VH (x86)
AX_CHECK_VE
################################

# Check MPI
AX_ENABLE_MPI

# Check OMP
AX_ENABLE_OPENMP

# Check CUpti GPU profiling interface
AX_ENABLE_CUDAPROF

# Check the OpenACC profiling interface
AX_ENABLE_ACCPROF

# Check PAPI hardware profiling support
AX_ENABLE_PAPI

# Check Likwid hardware profiling support
AX_ENABLE_LIKWID

# Check for dlopen
AX_ENABLE_DYNLIB_TRACING

# Enable hardware counters
#AX_ENABLE_HWC

# Check for Self-profiling
AX_ENABLE_SELF_PROFILING

# Check of libiberty is available
AX_WITH_LIBERTY

# Check consistency of HWC, Architecture, and Papi
#AX_CHECK_HWC_CONSISTENCY

LT_INIT(dlopen)
AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([src/Makefile])
AC_CONFIG_FILES([src/sorting/Makefile])
AC_CONFIG_FILES([src/logfiles/Makefile])
AC_CONFIG_FILES([src/mpi/Makefile])
AC_CONFIG_FILES([src/mpi/utils/Makefile])
AC_CONFIG_FILES([src/mpi/vftr_mpi/Makefile])
AC_CONFIG_FILES([src/mpi/c2vftr/Makefile])
AC_CONFIG_FILES([src/mpi/c-wrapper/Makefile])
AC_CONFIG_FILES([src/mpi/f2vftr/Makefile])
AC_CONFIG_FILES([src/mpi/f-wrapper/Makefile])
AC_CONFIG_FILES([src/mpi/f082vftr/Makefile])
AC_CONFIG_FILES([src/mpi/f08-wrapper/Makefile])
AC_CONFIG_FILES([src/omp/Makefile])
AC_CONFIG_FILES([src/cuda/Makefile])
AC_CONFIG_FILES([src/accprof/Makefile])
AC_CONFIG_FILES([src/hwprof/Makefile])
AC_CONFIG_FILES([external/Makefile])
AC_CONFIG_FILES([tools/Makefile])
AC_CONFIG_FILES([tools/self_profiling_analyzer/Makefile])
AC_CONFIG_FILES([tools/config_tools/Makefile])
AC_CONFIG_FILES([test/Makefile])
AC_CONFIG_FILES([test/units/Makefile])
AC_CONFIG_FILES([test/config/Makefile])
AC_CONFIG_FILES([test/user_functions/Makefile])
AC_CONFIG_FILES([test/omp/Makefile])
AC_CONFIG_FILES([test/cuda/Makefile])
AC_CONFIG_FILES([test/openacc/Makefile])
AC_CONFIG_FILES([test/hwprof/Makefile])
AC_CONFIG_FILES([test/mpi/Makefile])
AC_CONFIG_FILES([test/mpi/c-mpi/Makefile])
AC_CONFIG_FILES([test/mpi/f-mpi/Makefile])
AC_CONFIG_FILES([test/mpi/f08-mpi/Makefile])
AC_CONFIG_FILES([default_hw_scenarios/Makefile])

AC_OUTPUT

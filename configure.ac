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
AC_ENABLE_SHARED
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

# Check if fortran is selected or not
AX_ENABLE_FORTRAN

# Check if elf-symbols align with function pointers
AX_CHECK_VMAP_OFFSET

# Check if on VE or VH (x86)
AX_CHECK_VE
################################

# Check MPI
AX_CHECK_MPI

# Enable hardware counters
AX_ENABLE_HWC

# Enable OpenMP bounces
AX_ENABLE_OPENMP

# Check for Papi library
AX_PAPI_LIBRARY

# Check of libiberty is available
AX_LIBERTY

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
AC_CONFIG_FILES([external/Makefile])
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

AX_LINK_TEST_FILES

AC_OUTPUT

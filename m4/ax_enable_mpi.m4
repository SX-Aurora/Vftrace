# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_MPI
#
# DESCRIPTION
#
#   This macro checks if MPI support is wanted and available
#

AC_DEFUN([AX_ENABLE_MPI], [
   AC_PREREQ(2.50)
   AC_ARG_ENABLE([mpi],
      [AS_HELP_STRING([--enable-mpi], [enable MPI profiling layer. [default=no]])],
      [enable_mpi_present="yes"],
      [enable_mpi_present="no"])
   AC_MSG_CHECKING([whether MPI is enabled])
   # if the option is not given, resort to default (no)
   AS_IF([test "x$enable_mpi_present" = "xno"], [enable_mpi=no])
   AM_CONDITIONAL([ENABLE_MPI], [test "x$enable_mpi" = "xyes"])
   AC_MSG_RESULT([$enable_mpi])

   # NEC_MPI for x86 links mpi_mem and not mpi
   # Check which one is required and use that one
   # in further tests
   AM_COND_IF(
      [ENABLE_MPI],
      [AC_LANG(C)
       AC_CHECK_LIB(
          [mpi_mem],
          [MPI_Init],
          [mpi_lib_name=mpi_mem],
          [mpi_lib_name=mpi])]
       AC_MSG_NOTICE([Using -l${mpi_lib_name} for further testing!]))
   
   # check if compiler supports C-MPI
   AM_COND_IF(
      [ENABLE_MPI],
      [AC_LANG(C)
       AC_CHECK_LIB(
          [${mpi_lib_name}],
          [MPI_Init],
          [],
          [AC_MSG_FAILURE([unable to find C-MPI])])])

   # check if compiler supports F-MPI
   AM_COND_IF(
      [ENABLE_FORTRAN],
      [AM_COND_IF(
         [ENABLE_MPI],
         [AC_LANG(Fortran)
          AC_CHECK_LIB(
             [${mpi_lib_name}],
             [MPI_INIT],
             [],
             [AC_MSG_FAILURE([unable to find Fortran-MPI])])])])

   # Check for MPI-vendor
   AM_COND_IF([ENABLE_MPI], [
      # OpenMPI
      AC_MSG_CHECKING([whether OpenMPI is used])
      if test "x$(mpirun --version 2> /dev/null | grep "Open MPI" | wc -l)" = "x1" ; then
         uses_open_mpi="yes"
      else
         uses_open_mpi="no"
      fi
      AC_MSG_RESULT([$uses_open_mpi])])
   AM_CONDITIONAL([USES_OPEN_MPI],
                  [test "x$uses_open_mpi" = "xyes"])

   # Set the -D flag for the preprocessor globally
   AM_COND_IF([ENABLE_MPI], [
      AX_APPEND_FLAG([-D_MPI], [CFLAGS])
      AX_APPEND_FLAG([-D_MPI], [FCFLAGS])
      AX_APPEND_FLAG([-D_MPI], [CXXFLAGS])
      AX_APPEND_FLAG([-D_MPI], [CPPFLAGS])
      AX_APPEND_FLAG([-I$(realpath ${srcdir}/src/mpi/utils)], [CFLAGS])
      AX_APPEND_FLAG([-I$(realpath ${srcdir}/src/mpi/utils)], [FCFLAGS])
   ])

   # Check the OpenMPI version to determine proper oversubscribe-flag
   # With version 5 of OpenMPI a new flag was introduced
   # and the old one deprecated.
   # We need the flag as some mpi-tests require four processes
   # which can be problematic on systems with few cores
   AM_COND_IF([ENABLE_MPI], [
      AM_COND_IF([USES_OPEN_MPI], [
         AC_MSG_CHECKING([OpenMPI version])
         ompi_version="$(mpirun --version 2> /dev/null | head -n 1 | awk '{print $NF}')"
         AC_MSG_RESULT([${ompi_version}])
         AX_COMPARE_VERSION([${ompi_version}], [lt], [5.0.0],
            [ompi_version_lt5="yes"],
            [ompi_version_lt5="no"])])])
   AM_CONDITIONAL([OMPI_VERSION_LT5],
                  [test "x$ompi_version_lt5" = "xyes"])

   AM_COND_IF([ENABLE_MPI], [
      # NEC-MPI
      AC_MSG_CHECKING([whether NEC-MPI is used])
      if test "x$(mpirun --version 2> /dev/null | grep "NEC MPI" | wc -l)" = "x1" ; then
         uses_nec_mpi="yes"
      else
         uses_nec_mpi="no"
      fi
      AC_MSG_RESULT([$uses_nec_mpi])])
   AM_CONDITIONAL([USES_NEC_MPI],
                  [test "x$uses_nec_mpi" = "xyes"])

   AM_COND_IF([ENABLE_MPI], [
      # IntelMPI
      AC_MSG_CHECKING([whether IntelMPI is used])
      if test "x$(mpirun --version 2> /dev/null | grep "Intel MPI" | wc -l)" = "x1" ; then
         uses_intel_mpi="yes"
      else
         uses_intel_mpi="no"
      fi
      AC_MSG_RESULT([$uses_intel_mpi])])
   AM_CONDITIONAL([USES_INTEL_MPI],
                  [test "x$uses_intel_mpi" = "xyes"])

   # Check if Fortran-MPI supports TS29113
   # Fortran 90:
   AM_COND_IF(
      [ENABLE_FORTRAN],
      [AC_LANG(Fortran)
       AM_COND_IF(
          [ENABLE_MPI],
          [AC_MSG_CHECKING([whether MPI-F90 supports TS29113])
           AC_RUN_IFELSE(
             [AC_LANG_SOURCE([[
PROGRAM test
   USE mpi
   IF (MPI_SUBARRAYS_SUPPORTED) STOP 1
   STOP 0
END PROGRAM test]])],
             [supports_mpi_f90_ts29113=no],
             [supports_mpi_f90_ts29113=yes])
           AC_MSG_RESULT([$supports_mpi_f90_ts29113])]
   )])
   AM_CONDITIONAL([SUPPORTS_MPIF90_TS], [test "$supports_mpi_f90_ts29113" = "yes"])
   
   # Fortran 2008:
   AM_COND_IF(
      [ENABLE_FORTRAN],
      [AM_COND_IF(
         [ENABLE_MPI],
         [AC_MSG_CHECKING([whether MPI-F08 supports TS29113])
          AC_RUN_IFELSE(
            [AC_LANG_SOURCE([[
PROGRAM test
   USE mpi_f08
   IF (MPI_SUBARRAYS_SUPPORTED) STOP 1
   STOP 0
END PROGRAM test]])],
           [supports_mpi_f08_ts29113=no],
           [supports_mpi_f08_ts29113=yes])
       AC_MSG_RESULT([$supports_mpi_f08_ts29113])]
   )])
   AM_CONDITIONAL([SUPPORTS_MPIF08_TS], [test "$supports_mpi_f08_ts29113" = "yes"])

   # Check Supported MPI-Standard
   AX_CHECK_MPI_STD

   # Check for required profiling flags
   AX_CHECK_MPI_PROF_FLAGS
])

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

AC_DEFUN([AX_CHECK_MPI], [
   AC_PREREQ(2.50)
   AC_ARG_WITH([mpi],
      [AS_HELP_STRING([--with-mpi],
         [compile with MPI support. If none is found,
            MPI is not used. Default: no])],
      [with_mpi=yes])
   AM_CONDITIONAL([WITH_MPI], [test "$with_mpi" = "yes"])
   
   # check if compiler supports C-MPI
   AM_COND_IF(
      [WITH_MPI],
      [AC_LANG(C)
       AC_CHECK_LIB(
          [mpi],
          [MPI_Init],
          [],
          [AC_MSG_FAILURE([unable to find C-MPI])])])

   # check if compiler support F-MPI
   AM_COND_IF(
      [WITH_MPI],
      [AC_LANG(Fortran)
       AC_CHECK_LIB(
          [mpi],
          [MPI_INIT],
          [],
          [AC_MSG_FAILURE([unable to find Fortran-MPI])])])

   # Check for MPI-vendor
   AM_COND_IF([WITH_MPI], [
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

   AM_COND_IF([WITH_MPI], [
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

   AM_COND_IF([WITH_MPI], [
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
   AC_LANG(Fortran)
   AM_COND_IF(
      [WITH_MPI],
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
   )
   AM_CONDITIONAL([SUPPORTS_MPIF90_TS], [test "$supports_mpi_f90_ts29113" = "yes"])
   
   # Fortran 2008:
   AM_COND_IF(
      [WITH_MPI],
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
   )
   AM_CONDITIONAL([SUPPORTS_MPIF08_TS], [test "$supports_mpi_f08_ts29113" = "yes"])
])

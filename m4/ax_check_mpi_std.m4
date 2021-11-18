# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_MPI_STD
#
# DESCRIPTION
#
#   This macro checks which MPI standard is supported
#

AC_DEFUN([AX_CHECK_MPI_STD], [
   AC_PREREQ(2.50)
   AC_MSG_CHECKING([which MPI-standard])
   AC_LANG(C)
   for maj_ver in 1 2 3 4;
   do
      for min_ver in 1 2 3;
      do
         AC_RUN_IFELSE([
            AC_LANG_PROGRAM(
            [[#include <mpi.h>]],
            [[return !(MPI_VERSION == ${maj_ver} && MPI_SUBVERSION == ${min_ver});]])
         ],[
            mpi_version=${maj_ver}.${min_ver}
            break 2
         ],[
            mpi_version=none
         ])
      done
   done
   AC_MSG_RESULT([$mpi_version])
   AM_CONDITIONAL([VALID_MPI_VERSION],
                  [test "x$mpi_version" != "xnone"])
   AM_COND_IF([VALID_MPI_VERSION],
              [],
              [AC_MSG_ERROR([Unable to determine supported MPI-standard!])])
])

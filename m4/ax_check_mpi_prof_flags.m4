# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_MPI_PROF_FLAGS
#
# DESCRIPTION
#
#   This macro checks whether the compiler needs
#   special flags to activate the PMPI layer
#

AC_DEFUN([AX_CHECK_MPI_PROF_FLAGS], [
   AC_PREREQ(2.50)
   AM_COND_IF(
      [ENABLE_MPI],
      [AC_LANG(C)
       AX_CHECK_COMPILE_FLAG([-mpiprof],
            [AX_APPEND_FLAG([-mpiprof])])])])

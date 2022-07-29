# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_CXX_STDFLAG
#
# DESCRIPTION
#
#   This macro tries to find find an appropriate std-flag for the used cxx-compiler
#

AC_DEFUN([AX_CHECK_CXX_STDFLAG], [
   AC_LANG(C++)
   AC_PREREQ(2.50)
   AX_CHECK_COMPILE_FLAG([-stdlib=libc++],
      [AX_APPEND_FLAG([-stdlib=libc++])])])

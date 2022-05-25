# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_C99_STD
#
# DESCRIPTION
#
#   This macro tries to find find an appropriate std-flag for the used c-compiler
#   such that the C99 standard is supported.
#

AC_DEFUN([AX_CHECK_C99_STD], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AX_CHECK_COMPILE_FLAG([-std=gnu11],
      [AX_APPEND_FLAG([-std=gnu11])],
      [AX_CHECK_COMPILE_FLAG([-std=c11],
         [AX_APPEND_FLAG([-std=c11])],
         [AX_CHECK_COMPILE_FLAG([-std=c99],
            [AX_APPEND_FLAG([-std=c99])],
            [AC_MSG_ERROR([C compiler does not support at least C99!])])])])])

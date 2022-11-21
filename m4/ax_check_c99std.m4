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

AC_DEFUN([AX_CHECK_STD_FLAG],
   [AM_COND_IF([$1],
      [AX_CHECK_COMPILE_FLAG([$2],
         [AX_APPEND_FLAG([-std=gnu11])
          AM_CONDITIONAL([$1], [test "false" = "true"])],
         [AM_CONDITIONAL([$1], [test "true" = "true"])])])])

AC_DEFUN([AX_CHECK_C99_STD], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AM_CONDITIONAL([HAS_NO_C99_STD], [test "true" = "true"])

   AX_CHECK_STD_FLAG([HAS_NO_C99_STD], [-std=gnu11])
   AX_CHECK_STD_FLAG([HAS_NO_C99_STD], [--std=gnu11])

   AX_CHECK_STD_FLAG([HAS_NO_C99_STD], [-std=c11])
   AX_CHECK_STD_FLAG([HAS_NO_C99_STD], [--std=c11])

   AX_CHECK_STD_FLAG([HAS_NO_C99_STD], [-std=c99])
   AX_CHECK_STD_FLAG([HAS_NO_C99_STD], [--std=c99])

   AM_COND_IF([HAS_NO_C99_STD],
      [AC_MSG_ERROR([C compiler does not support at least C99!])])])

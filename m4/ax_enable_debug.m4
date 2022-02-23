# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_ENABLE_DEBUG
#
# DESCRIPTION
#
#   This macro check for enabled debugging output
#

AC_DEFUN([AX_ENABLE_DEBUG], [
   AC_PREREQ(2.50)
   AC_ARG_ENABLE(
      [debug],
      [AS_HELP_STRING([--enable-debug],
                      [build with additional debugging code [default=no]])],
      [],
      [enable_debug="no"])
   AM_CONDITIONAL([ENABLE_DEBUG], [test "$enable_debug" = yes])
   AM_COND_IF(
      [ENABLE_DEBUG],
      [AX_APPEND_FLAG([-g])
       AX_APPEND_FLAG([-D_DEBUG])])

])

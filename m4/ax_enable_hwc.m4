# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_ENABLE_HWC
#
# DESCRIPTION
#
#   This macro checks for hardware counters
#

AC_DEFUN([AX_ENABLE_HWC], [
   AC_PREREQ(2.50)
   AC_ARG_ENABLE(
      [hwc],
      [AS_HELP_STRING([--enable-hwc], [enable hadware counter [default=no]])],
      [],
      [enable_hwc="no"])
   AM_CONDITIONAL([ENABLE_HWC], [test "$enable_hwc" = "yes"])
])

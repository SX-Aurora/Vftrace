# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_UINTPTR
#
# DESCRIPTION
#
#   This macro check whether the c-compiler supports the uintptr_t type
#

AC_DEFUN([AX_CHECK_UINTPTR], [
AC_PREREQ(2.50)
AC_CHECK_TYPE([uintptr_t],
   [AX_APPEND_FLAG([-D_HAS_UINTPTR])],
   [AX_APPEND_FLAG([-U_HAS_UINTPTR])],
   [[#include <stdint.h>]])
])

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
   AC_MSG_CHECKING([whether system libc++ can be used])
   AC_RUN_IFELSE([
     AC_LANG_SOURCE([[
#include <iostream>
main() {return 1};
]])]
   [AX_APPEND_FLAG([-stdlib=libc++])
   ][])
])

# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_VE
#
# DESCRIPTION
#
#   This macro checks if the code is compiled for a VE
#

AC_DEFUN([AX_CHECK_VE], [
   AC_PREREQ(2.50)
   AC_LANG(C)
      AC_MSG_CHECKING([if on vector engine])
      AC_RUN_IFELSE(
        [AC_LANG_SOURCE([[
main() {
#ifdef __ve__
   return 0;
#else
   return 1;
#endif
}
        ]])],
         [on_vector_engine=yes],
         [on_vector_engine=no])
   AM_CONDITIONAL([ON_VECTOR_ENGINE], [test "$on_vector_engine" = "yes"])
   AC_MSG_RESULT([$on_vector_engine])
   
])

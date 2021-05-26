# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_LIBERTY
#
# DESCRIPTION
#
#   This macro checks if the library libiberty is properly included
#
AC_DEFUN([AX_LIBERTY], [
   AC_LANG(C)
   AC_MSG_CHECKING([if libiberty is available])
   AC_CHECK_LIB([iberty], [cplus_demangle], [has_liberty=yes] ,[AC_MSG_FAILURE([unable to find libiberty])])
   
   AM_CONDITIONAL([HAS_LIBERTY], [test "$has_liberty" = "yes"])
   AM_COND_IF([HAS_LIBERTY],
     AC_CHECK_HEADER([demangle.h], [has_demangle=yes],[AC_MSG_FAILURE([unable to find demangle.h])]), )
   AM_CONDITIONAL([HAS_LIBERTY], [test "$has_demangle" = "yes"])
])


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
   AC_CHECK_LIB([iberty],
                [cplus_demangle],
                [has_liberty=yes],
                [has_liberty=no])
   AM_CONDITIONAL([HAS_LIBERTY], [test "x$has_liberty" = "xyes"])
   AM_COND_IF([HAS_LIBERTY],
              [AC_CHECK_HEADER([demangle.h],
                               [has_demangle=yes],
                               [has_demangle=no])])
   AM_CONDITIONAL([HAS_LIBERTY], [test "x$has_demangle" = "xyes"])
   AM_COND_IF([HAS_LIBERTY], [AX_APPEND_FLAG([-liberty], [LDFLAGS])])
])


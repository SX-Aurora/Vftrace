# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_PAPI_LIBRARY
#
# DESCRIPTION
#
#   This macro checks if the papi library is properly included
#

AC_DEFUN([AX_PAPI_LIBRARY], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AC_ARG_WITH([papi_inc],
               [AC_HELP_STRING([--with-papi-inc=DIR],
                               [papi include directory])],
               [
               CFLAGS="-I${withval} ${CFLAGS}"
               has_papi=yes])
   AC_ARG_WITH([papi_lib],
               [AC_HELP_STRING([--with-papi-lib=DIR],
                               [papi lib directory])],
               [
               LDFLAGS="-L$withval $LDFLAGS"
               has_papi=yes])
   AC_ARG_WITH([papi],
               [AC_HELP_STRING([--with-papi=DIR],
                               [papi install directory])],
               [
               if test -z "$with_papi_inc" -a -z "$with_papi_lib" ; then
                  CPPFLAGS="-I$withval/include $CPPFLAGS"
                  LDFLAGS="-L$withval/lib -lpapi $LDFLAGS"
                  has_papi=yes
               else
                  AC_MSG_FAILURE([Do not use --with-papi and --with-papi-inc/--with-papi-lib options simultaneously.])
               fi],
               [
               if test -z "$with_papi_inc" -a -z "$with_papi_lib" ; then
                   has_papi=no
               fi])
   AM_CONDITIONAL([HAS_PAPI], [test "$has_papi" = "yes"])
   AM_COND_IF([HAS_PAPI],
      [AC_CHECK_LIB([papi],
          [PAPI_library_init],
          [],
          [AC_MSG_FAILURE([unable to find papi library])])])
])

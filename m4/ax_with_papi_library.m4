# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_WITH_PAPI_LIBRARY
#
# DESCRIPTION
#
#   This macro checks if the papi library is properly included
#

AC_DEFUN([AX_WITH_PAPI_LIBRARY], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AC_ARG_WITH(
      [papi],
      [AC_HELP_STRING([--with-papi=DIR], [papi install directory])],
      [with_papi_present="yes"],
      [with_papi_present="no"])
   # if the option is not given, resort to default (no)
   AS_IF([test "x$with_papi_present" = "xno"], [with_papi="no"])

   AC_ARG_WITH(
      [papi_inc],
      [AC_HELP_STRING([--with-papi-inc=DIR], [papi include directory])],
      [with_papi_inc_present="yes"],
      [with_papi_inc_present="no"])
   # if the option is not given, resort to default (no)
   AS_IF([test "x$with_papi_inc_present" = "xno"], [with_papi_inc="no"])

   AC_ARG_WITH(
      [papi_lib],
      [AC_HELP_STRING([--with-papi-lib=DIR], [papi lib directory])],
      [with_papi_lib_present="yes"],
      [with_papi_lib_present="no"])
   # if the option is not given, resort to default (no)
   AS_IF([test "x$with_papi_lib_present" = "xno"], [with_papi_lib="no"])

   # Check if any of the above options were selected
   AS_IF([test "x$with_papi_present" = "xyes" || test "x$with_papi_inc_present" = "xyes" || test "x$with_papi_lib_present" = "xyes"], [has_papi="yes"], [has_papi="no"])
   AM_CONDITIONAL([HAS_PAPI], [test "x$has_papi" = "xyes"])

   AM_COND_IF(
      [HAS_PAPI],
      [# If path are given add them to
       # CFLAGS as include-paths, and
       # LDFLAGS as library-path respectively
       AS_IF([test "x$with_papi_present" = "xyes" && test "x$with_papi" != "xyes" && test "x$with_papi" != "xno"],
          [AX_APPEND_FLAG([-I${with_papi}/include/], [CFLAGS])
           AX_APPEND_FLAG([-L${with_papi}/lib/], [LDFLAGS])])
       AS_IF([test "x$with_papi_inc_present" = "xyes" && test "x$with_papi_inc" != "xyes" && test "x$with_papi_inc" != "xno"],
          [AX_APPEND_FLAG([-I${with_papi_inc}/], [CFLAGS])])
       AS_IF([test "x$with_papi_lib_present" = "xyes" && test "x$with_papi_lib" != "xyes" && test "x$with_papi_lib" != "xno"],
          [AX_APPEND_FLAG([-L${with_papi_lib}/], [LDFLAGS])])
       AX_APPEND_FLAG([-lpapi], [LDFLAGS])])

   # Check whether the given path are sufficient to find the header and library
   AM_COND_IF([HAS_PAPI],
      [AC_CHECK_LIB([papi],
          [PAPI_library_init], ,
          [AC_MSG_FAILURE([unable to find papi library])])])
   AM_COND_IF([HAS_PAPI],
      [AC_CHECK_HEADER([papi.h], ,
          [AC_MSG_FAILURE([unable to find papi headers])])])
])

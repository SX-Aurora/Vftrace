# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_WITH_LIBERTY
#
# DESCRIPTION
#
#   This macro check for libiberty
#

AC_DEFUN([AX_WITH_LIBERTY], [
   AC_PREREQ(2.50)
   AC_ARG_WITH(
      [liberty],
      [AS_HELP_STRING([--with-liberty[=DIR]], [libiberty install directory for C++ symbol demangling [default=no]])],
      [],
      [with_liberty="no"])
   AC_ARG_WITH(
      [liberty_inc],
      [AS_HELP_STRING([--with-liberty_inc=DIR], [libiberty include directory for C++ symbol demangling [default=no]])],
      [],
      [with_liberty_inc="no"])
   AC_ARG_WITH(
      [liberty_lib],
      [AS_HELP_STRING([--with-liberty_lib=DIR], [libiberty lib directory for C++ symbol demangling [default=no]])],
      [],
      [with_liberty_lib="no"])

   AM_CONDITIONAL([WITH_LIBERTY], [test "x$with_liberty" != "xno"])
   AM_CONDITIONAL([WITH_LIBERTY_INC], [test "x$with_liberty_inc" != "xno"])
   AM_CONDITIONAL([WITH_LIBERTY_LIB], [test "x$with_liberty_lib" != "xno"])

   AC_MSG_CHECKING([whether to use liberty for C++ symbol demangling])
   AM_COND_IF([WITH_LIBERTY], [use_liberty="yes"], [use_liberty="no"])
   AM_COND_IF([WITH_LIBERTY_INC], [use_liberty="yes"])
   AM_COND_IF([WITH_LIBERTY_LIB], [use_liberty="yes"])
   AC_MSG_RESULT([$use_liberty])
   AM_CONDITIONAL([USE_LIBERTY], [test "x$use_liberty" = "xyes"])

   AM_COND_IF([WITH_LIBERTY],
      [AS_IF([test "x$with_liberty" != "xyes"],
         [with_liberty_has_path="yes"],
         [with_liberty_has_path="no"])])
   AM_CONDITIONAL([WITH_LIBERTY_HAS_PATH],
      [test "x$with_liberty_has_path" = "xyes"])
   AM_COND_IF([WITH_LIBERTY_INC],
      [AS_IF([test "x$with_liberty_inc" != "xyes"],
         [with_liberty_inc_has_path="yes"],
         [with_liberty_inc_has_path="no"])])
   AM_CONDITIONAL([WITH_LIBERTY_INC_HAS_PATH],
      [test "x$with_liberty_inc_has_path" = "xyes"])
   AM_COND_IF([WITH_LIBERTY_LIB],
      [AS_IF([test "x$with_liberty_lib" != "xyes"],
         [with_liberty_lib_has_path="yes"],
         [with_liberty_lib_has_path="no"])])
   AM_CONDITIONAL([WITH_LIBERTY_LIB_HAS_PATH],
      [test "x$with_liberty_lib_has_path" = "xyes"])

   # Set include path
   AM_COND_IF([WITH_LIBERTY_HAS_PATH], [
      AX_APPEND_FLAG([-I${with_liberty}/include/], [CFLAGS])
      AX_APPEND_FLAG([-I${with_liberty}/include/], [CPPFLAGS])])
   AM_COND_IF([WITH_LIBERTY_INC_HAS_PATH], [
      AX_APPEND_FLAG([-I${with_liberty_inc}], [CFLAGS])
      AX_APPEND_FLAG([-I${with_liberty_inc}], [CPPFLAGS])])

   # Set library path
   AM_COND_IF([WITH_LIBERTY_HAS_PATH], [
      AX_APPEND_FLAG([-L${with_liberty}/lib/], [LDFLAGS])])
   AM_COND_IF([WITH_LIBERTY_LIB_HAS_PATH], [
      AX_APPEND_FLAG([-L${with_liberty_lib}], [LDFLAGS])])
      
   AM_COND_IF([USE_LIBERTY], [
      AC_CHECK_HEADER([demangle.h],
                      [has_demangle=yes],
                      [has_demangle=no])])
   AM_CONDITIONAL([HAS_LIBERTY], [test "x$has_demangle" = "xyes"])
   AM_COND_IF([USE_LIBERTY], [
      AM_COND_IF([HAS_LIBERTY],
         [],
         [AC_MSG_ERROR([cannot find headers for liberty])])])


   AM_COND_IF([USE_LIBERTY], [
      AC_CHECK_LIB([iberty],
                   [cplus_demangle],
                   [has_liberty=yes],
                   [has_liberty=no])])
   AM_CONDITIONAL([HAS_LIBERTY], [test "x$has_liberty" = "xyes"])
   AM_COND_IF([USE_LIBERTY], [
      AM_COND_IF([HAS_LIBERTY],
         [AX_APPEND_FLAG([-liberty], [LDFLAGS])],
         [AC_MSG_ERROR([cannot find -liberty])])])
])

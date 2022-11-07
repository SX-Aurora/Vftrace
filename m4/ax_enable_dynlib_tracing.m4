# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_ENABLE_DYNLIB_TRACING
#
# DESCRIPTION
#
#   This macro checks if tracing of libraried loaded via dlopen should be enabled
#   This clashes with cupti as both vftrace and cupti overwrite dlopen functionality.
#

AC_DEFUN([AX_ENABLE_DYNLIB_TRACING], [
   AC_PREREQ(2.50)
   AC_ARG_ENABLE([dynlib-tracing],
      [AS_HELP_STRING([--enable-dynlib-tracing],
                      [enable tracing of library functions loaded with dlopen.
                      Conflicts with cupti usage. [default=no]])],
      [enable_dynlib_tracing_present="yes"],
      [enable_dynlib_tracing_present="no"])
   AC_MSG_CHECKING([whether dynamic library tracing is enabled])
   # if the option is not given, resort to default (no)
   AS_IF([test "x$enable_dynlib_tracing_present" = "xno"], [enable_dynlib_tracing=no])
   AM_CONDITIONAL([ENABLE_DYNLIB_TRACING], [test "x$enable_dynlib_tracing" = "xyes"])
   AC_MSG_RESULT([$enable_dynlib_tracing])

   # check if dlopen and cupti are activated at the same time
   # abort configure if it is the case
   AM_COND_IF([ENABLE_DYNLIB_TRACING],
      [AM_COND_IF([ENABLE_CUDAPROF],
          [AC_MSG_FAILURE(
             [dynamic library tracing and cupti cannot be active simultaneously!])])])
])

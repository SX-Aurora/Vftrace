# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_ENABLE_SELF_PROFILING
#
# DESCRIPTION
#
#   This macro checks if vftrace's self profiling capabilities
#   are supported and should be compiled in
#

AC_DEFUN([AX_ENABLE_SELF_PROFILING], [
   AC_PREREQ(2.50)
   AC_ARG_ENABLE([self_profiling],
      [AS_HELP_STRING([--enable-self-profiling],
                      [enable self-profiling of vftrace. Only for vftrace-developers [default=no]])],
      [enable_self_profiling_present="yes"],
      [enable_self_profiling_present="no"])
   AC_MSG_CHECKING([whether self profiling is enabled])
   # if the option is not given, resort to default (no)
   AS_IF([test "x$enable_self_profiling_present" = "xno"], [enable_self_profiling=no])
   AM_CONDITIONAL([ENABLE_SELF_PROFILING], [test "x$enable_self_profiling" = "xyes"])
   AC_MSG_RESULT([$enable_self_profiling])

   # check if preprocessor supports the __FUNCTION__ variable
   AM_COND_IF(
      [ENABLE_SELF_PROFILING],
      [AC_LANG(C)
        AC_MSG_CHECKING([whether __FUNCTION__ variable is supported])
        AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
        int main() {
           __FUNCTION__;
        }
        ]])],[function_var_supported="yes"],[function_var_supported="no"])
        AC_MSG_RESULT([$function_var_supported])])
   AM_CONDITIONAL([SELF_PROFILING],
                  [test "x$function_var_supported" = "xyes"])
   #TODO check fortran
])

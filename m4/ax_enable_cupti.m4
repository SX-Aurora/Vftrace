AC_DEFUN([AX_ENABLE_CUPTI], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AC_ARG_WITH([cupti],
               [AC_HELP_STRING([--with-cupti=DIR],
                               [CUPTI installation directory. Enables CUDA or OpenCL profiling.])],
               [
               if test "x$with_cupti" = "xyes" ; then
                  if test -z "$with_cupti_lib" ; then
                     LDFLAGS="-lcupti $LDFLAGS"
                  fi
               else
                  if test -z "$with_cupti_inc" ; then
                     CFLAGS="-I$withval/include $CPPFLAGS"
                  fi
                  if test -z "$with_cupti_lib" ; then
                     LDFLAGS="-L$withval/lib64 -lcupti $LDFLAGS"
                  fi
               fi
               enable_cupti=yes
               ],
               [
               if test -z "$with_cupti_inc" -a -z "$with_cupti_lib" ; then
                  enable_cupti=no
               fi])
  
   AM_CONDITIONAL([ENABLE_CUPTI], [test "$enable_cupti" = "yes"])
   AM_COND_IF([ENABLE_CUPTI],
      [AC_CHECK_LIB([cupti],
          [cuptiSubscribe], ,
          [AC_MSG_FAILURE([unable to find CUPTI library])])])
   AM_COND_IF([ENABLE_CUPTI],
      [AC_CHECK_HEADER([cupti.h], ,
          [AC_MSG_FAILURE([unable to find CUPTI headers])])])
])

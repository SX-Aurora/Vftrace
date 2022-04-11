AC_DEFUN([AX_CUPTI], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AC_ARG_WITH([cupti_inc],
               [AC_HELP_STRING([--with-cupti-inc=DIR],
                               [CUPTI include directory])],
               [
               CFLAGS="-I${withval} ${CFLAGS}"
               has_cupti=yes])
   AC_ARG_WITH([cupti_lib],
               [AC_HELP_STRING([--with-cupti-lib=DIR],
                               [CUPTI lib directory])],
               [
               LDFLAGS="-L${withval} -lcupti $LDFLAGS"
               has_cupti=yes])
   AC_ARG_WITH([cupti],
               [AC_HELP_STRING([--with-cupti=DIR],
                               [CUPTI INSTALL DIRECTORY])],
               [
               if test "x$with_cupti" = "xyes" ; then
                  if test -z "$with_cupti_lib" ; then
                     LDFLAGS="-lcupti $LDFLAGS"
                  fi
               else
                  if test -z "$with_cupti_inc" ; then
                     CPPFLAGS="-I$withval/include $CPPFLAGS"
                  fi
                  if test -z "$with_cupti_lib" ; then
                     LDFLAGS="-L$withval/lib64 -lcupti $LDFLAGS"
                  fi
               fi
               has_cupti=yes
               ],
               [
               if test -z "$with_cupti_inc" -a -z "$with_cupti_lib" ; then
                  has_cupti=no
               fi])
   AM_CONDITIONAL([HAS_CUPTI], [test "$has_cupti" = "yes"])
   AM_COND_IF([HAS_CUPTI],
      [AC_CHECK_LIB([cupti],
          [cuptiSubscribe], ,
          [AC_MSG_FAILURE([unable to find CUPTI library])])])
   AM_COND_IF([HAS_CUPTI],
      [AC_CHECK_HEADER([cupti.h], ,
          [AC_MSG_FAILURE([unable to find CUPTI headers])])])
])

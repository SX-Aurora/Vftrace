AC_DEFUN([AX_ENABLE_VEDA], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AC_ARG_WITH([veda],
               [AC_HELP_STRING([--with-veda=DIR],
                               [VEDA installation directory. Enables VEDA profiling.])],

	       [with_veda_present="yes"],
	       [with_veda_present="no"]
   )
  
   AC_MSG_CHECKING([whether VEDA is supported])
   AM_CONDITIONAL([ENABLE_VEDA], [test "$with_veda_present" = "yes"])
   AC_MSG_RESULT(${with_veda_present})
   AM_COND_IF([ENABLE_VEDA],
      [AX_APPEND_FLAG([-L${with_veda}/lib64], [LDFLAGS])])
   AM_COND_IF([ENABLE_VEDA],
       [AX_APPEND_FLAG([-lveda], [LDFLAGS])])
   AM_COND_IF([ENABLE_VEDA],
       [AX_APPEND_FLAG([-I${with_veda}/include], [CFLAGS])])
   AM_COND_IF([ENABLE_VEDA],
       [AX_APPEND_FLAG([-I${with_veda}/include], [CPPFLAGS])])
   AM_COND_IF([ENABLE_VEDA],
      [AC_CHECK_LIB([veda],
          [vedaProfilerSetCallback],
          [AC_MSG_FAILURE([unable to find VEDA library])])])
   AM_COND_IF([ENABLE_VEDA],
      [AC_CHECK_HEADER([veda.h],
          [AC_MSG_FAILURE([unable to find VEDA headers])])])
])

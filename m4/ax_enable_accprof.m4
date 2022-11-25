AC_DEFUN([AX_ENABLE_ACCPROF], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AC_ARG_WITH([accprof],
               [AC_HELP_STRING([--with-accprof=DIR],
                               [OpenACC installation directory containing libaccprof. Enables OpenACC profiling.])],

	       [with_accprof_present="yes"],
	       [with_accprof_present="no"]
   )
  
   AC_MSG_CHECKING([whether accprof is supported])
   AM_CONDITIONAL([ENABLE_ACCPROF], [test "$with_accprof_present" = "yes"])

   AC_MSG_CHECKING([if NVIDIA compiler is used])
   if test "x$($CC --version | grep -m 1 -i nvidia | wc -l)" = "x1" ; then
      has_nvidia_compiler="yes"
   else
      has_nvidia_compiler="no"
   fi
   AC_MSG_RESULT([$has_nvidia_compiler])

   AM_CONDITIONAL([HAS_NVIDIA_COMPILER], [test "x$has_nvidia_compiler" = "xyes"])
   AM_COND_IF([ENABLE_ACCPROF],
      [AM_COND_IF([HAS_NVIDIA_COMPILER], [], [AC_MSG_FAILURE([Need NVIDIA compiler])])], [], [])

   AC_MSG_RESULT(${with_accprof_present})
   AM_COND_IF([ENABLE_ACCPROF],
      [AX_APPEND_FLAG([-L${with_accprof}/lib], [LDFLAGS])])
   AM_COND_IF([ENABLE_ACCPROF],
       [AX_APPEND_FLAG([-I${with_accprof}/include], [CFLAGS])])
   AM_COND_IF([ENABLE_ACCPROF],
       [AX_APPEND_FLAG([-I${with_accprof}/include], [CPPFLAGS])])
   AM_COND_IF([ENABLE_ACCPROF],
      [AC_CHECK_HEADER([acc_prof.h], ,
          [AC_MSG_FAILURE([unable to find accprof headers])])])
])

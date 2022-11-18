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
   AC_MSG_RESULT(${with_accprof_present})
   AM_COND_IF([ENABLE_ACCPROF],
      [AX_APPEND_FLAG([-L${with_accprof}/lib], [LDFLAGS])])
   AM_COND_IF([ENABLE_ACCPROF],
      [AX_APPEND_FLAG([-laccprof -lnvc -lm -lnvcpumath -ldl], [LDFLAGS])])
   AM_COND_IF([ENABLE_ACCPROF],
       [AX_APPEND_FLAG([-I${with_accprof}/include], [CFLAGS])])
   AM_COND_IF([ENABLE_ACCPROF],
       [AX_APPEND_FLAG([-I${with_accprof}/include], [CPPFLAGS])])
   AM_COND_IF([ENABLE_ACCPROF],
       [AC_CHECK_LIB([accprof],
           [acc_register_library], ,
       	   [AC_MSG_FAILURE([unable to find accprof library])])])
   AM_COND_IF([ENABLE_ACCPROF],
      [AC_CHECK_HEADER([acc_prof.h], ,
          [AC_MSG_FAILURE([unable to find accprof headers])])])
])

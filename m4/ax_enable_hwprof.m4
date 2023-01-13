AC_DEFUN([AX_ENABLE_HWPROF], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AC_ARG_ENABLE([hwprof],
               [AC_HELP_STRING([--enable-hwprof],
                               [Enables hardware profiling with either PAPI or veperf.])],
               [enable_hwprof_present="yes"],
 	       [enable_hwprof_present="no"]
   )
   AC_MSG_CHECKING([whether hardware profiling is used])
   AM_CONDITIONAL([ENABLE_HWPROF], [test "$enable_hwprof_present" = "yes"])
   AC_MSG_RESULT(${enable_hwprof_present})

   # TODO: Check if we are on SX Aurora
])

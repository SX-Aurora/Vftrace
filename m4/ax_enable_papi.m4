AC_DEFUN([AX_ENABLE_PAPI], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AC_ARG_WITH([papi],
               [AC_HELP_STRING([--with-papi=DIR],
                               [PAPI installation directory. Enables PAPI hardware profiling.])],
               [with_papi_present="yes"],
 	       [with_papi_present="no"]
   )
   AC_MSG_CHECKING([whether PAPI is supported])
   AM_CONDITIONAL([ENABLE_PAPI], [test "$with_papi_present" = "yes"])

   AC_MSG_RESULT(${with_papi_present})
   AM_COND_IF([ENABLE_PAPI],
      [AX_APPEND_FLAG([-L${with_papi}/lib], [LDFLAGS])])
   AM_COND_IF([ENABLE_PAPI],
      [AX_APPEND_FLAG([-lpapi], [LDFLAGS])])
   AM_COND_IF([ENABLE_PAPI],
      [AX_APPEND_FLAG([-I${with_papi}/include], [CFLAGS])])
   AM_COND_IF([ENABLE_PAPI],
      [AX_APPEND_FLAG([-I${with_papi}/include], [CPPFLAGS])])
   AM_COND_IF([ENABLE_PAPI],
      [AC_CHECK_LIB([papi],
          [PAPI_library_init], ,
          [AC_MSG_FAILURE([unable to find PAPI library])])])
   AM_COND_IF([ENABLE_PAPI],
      [AC_CHECK_HEADER([papi.h], ,
          [AC_MSG_FAILURE([unable to find PAPI headers])])]) 
])

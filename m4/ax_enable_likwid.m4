AC_DEFUN([AX_ENABLE_LIKWID], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AC_ARG_WITH([likwid],
               [AC_HELP_STRING([--with-likwid=DIR],
                               [Likwid installation directory.])],
               [with_likwid_present="yes"],
 	       [with_likwid_present="no"]
   )
   AC_MSG_CHECKING([whether Likwid is supported])
   AM_CONDITIONAL([ENABLE_LIKWID], [test "$with_likwid_present" = "yes"])

   AC_MSG_RESULT(${with_likwid_present})
   AM_COND_IF([ENABLE_LIKWID],
      [AX_APPEND_FLAG([-L${with_likwid}/lib], [LDFLAGS])])
   AM_COND_IF([ENABLE_LIKWID],
      [AX_APPEND_FLAG([-llikwid], [LDFLAGS])])
   AM_COND_IF([ENABLE_LIKWID],
      [AX_APPEND_FLAG([-I${with_likwid}/include], [CFLAGS])])
   AM_COND_IF([ENABLE_LIKWID],
      [AX_APPEND_FLAG([-I${with_likwid}/include], [CPPFLAGS])])
   AM_COND_IF([ENABLE_LIKWID],
      [AC_CHECK_LIB([likwid],
          [topology_init], ,
          [AC_MSG_FAILURE([unable to find Likwid library])])])
   AM_COND_IF([ENABLE_LIKWID],
      [AC_CHECK_HEADER([likwid.h], ,
          [AC_MSG_FAILURE([unable to find Likwid headers])])]) 
])

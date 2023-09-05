AC_DEFUN([AX_ENABLE_CUDAPROF], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AC_ARG_WITH([cupti],
               [AC_HELP_STRING([--with-cupti=DIR],
                               [CUpti installation directory. Enables CUDA profiling.])],

	       [with_cupti_present="yes"],
	       [with_cupti_present="no"]
   )
   AC_MSG_CHECKING([whether Cupti is supported])
   AM_CONDITIONAL([ENABLE_CUDAPROF], [test "$with_cupti_present" = "yes"])

   AC_MSG_RESULT(${with_cupti_present})
   AM_COND_IF([ENABLE_CUDAPROF],
      [AX_APPEND_FLAG([-L${with_cupti}/lib64], [LDFLAGS])])
   AM_COND_IF([ENABLE_CUDAPROF],
       [AX_APPEND_FLAG([-lcupti], [LDFLAGS])])
   AM_COND_IF([ENABLE_CUDAPROF],
       [AX_APPEND_FLAG([-I${with_cupti}/include], [CFLAGS])])
   AM_COND_IF([ENABLE_CUDAPROF],
       [AX_APPEND_FLAG([-I${with_cupti}/include], [CPPFLAGS])])
   AM_COND_IF([ENABLE_CUDAPROF],
      [AC_CHECK_LIB([cupti],
          [cuptiSubscribe], ,
          [AC_MSG_FAILURE([unable to find CUpti library])])])
   AM_COND_IF([ENABLE_CUDAPROF],
      [AC_CHECK_HEADER([cupti.h], ,
          [AC_MSG_FAILURE([unable to find CUpti headers])])])
   AM_COND_IF([ENABLE_CUDAPROF],
      [AC_CHECK_LIB([cudart],
          [cudaGetDeviceCount], ,
          [AC_MSG_FAILURE([unable to find CudaRT library])])])
])

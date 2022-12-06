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

   AC_MSG_CHECKING([if NVIDIA compiler is used])
   if test "x$($CC --version | grep -m 1 -i nvidia | wc -l)" = "x1" ; then
      has_nvidia_compiler="yes"
   else
      has_nvidia_compiler="no"
   fi
   AC_MSG_RESULT([$has_nvidia_compiler])

   AM_CONDITIONAL([HAS_NVIDIA_COMPILER], [test "x$has_nvidia_compiler" = "xyes"])
   AM_COND_IF([ENABLE_CUDAPROF],
      [AM_COND_IF([HAS_NVIDIA_COMPILER], [], [AC_MSG_FAILURE([Need NVIDIA compiler])])], [], [])

   AC_MSG_RESULT(${with_cupti_present})

   # Set the -D flag for the preprocessor globally
   AM_COND_IF([ENABLE_CUDAPROF], [
      AX_APPEND_FLAG([-D_CUDA], [CFLAGS])
      AX_APPEND_FLAG([-D_CUDA], [CPPFLAGS])
      AX_APPEND_FLAG([-I$(realpath ${srcdir}/src/cuda)], [CFLAGS])
   ])

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

# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_SIZEOF_TYPES
#
# DESCRIPTION
#
#   This macro evaluates a set of sizeoffs for differenc C-types
#

AC_DEFUN([AX_SIZEOF_TYPES], [
AC_PREREQ(2.50)
# integers
   AC_CHECK_SIZEOF([char])
   AC_CHECK_SIZEOF([unsigned char])
   AC_CHECK_SIZEOF([short])
   AC_CHECK_SIZEOF([unsigned short])
   AC_CHECK_SIZEOF([int])
   AC_CHECK_SIZEOF([unsigned int])
   AC_CHECK_SIZEOF([long])
   AC_CHECK_SIZEOF([unsigned long])
   AC_CHECK_SIZEOF([long long])
   AC_CHECK_SIZEOF([unsigned long long])

   AC_CHECK_SIZEOF([int8_t], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([uint8_t], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([int16_t], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([uint16_t], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([int32_t], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([uint32_t], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([int64_t], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([uint64_t], [], [#include <inttypes.h>])

# floating point
   AC_CHECK_SIZEOF([float])
   AC_CHECK_SIZEOF([double])
   AC_CHECK_SIZEOF([long double])

# miscellaneous
   AC_CHECK_SIZEOF([bool], [], [#include <stdbool.h>])
   AC_CHECK_SIZEOF([void], [], [#include <stdlib.h>])

# integer pointer
   AC_CHECK_SIZEOF([char *])
   AC_CHECK_SIZEOF([unsigned char *])
   AC_CHECK_SIZEOF([short *])
   AC_CHECK_SIZEOF([unsigned short *])
   AC_CHECK_SIZEOF([int *])
   AC_CHECK_SIZEOF([unsigned int *])
   AC_CHECK_SIZEOF([long *])
   AC_CHECK_SIZEOF([unsigned long *])
   AC_CHECK_SIZEOF([long long *])
   AC_CHECK_SIZEOF([unsigned long long *])

   AC_CHECK_SIZEOF([int8_t *], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([uint8_t *], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([int16_t *], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([uint16_t *], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([int32_t *], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([uint32_t *], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([int64_t *], [], [#include <inttypes.h>])
   AC_CHECK_SIZEOF([uint64_t *], [], [#include <inttypes.h>])

# floating point pointer
   AC_CHECK_SIZEOF([float *])
   AC_CHECK_SIZEOF([double *])
   AC_CHECK_SIZEOF([long double *])

# miscellaneous pointer
   AC_CHECK_SIZEOF([bool *], [], [#include <stdbool.h>])
   AC_CHECK_SIZEOF([void *], [], [#include <stdlib.h>])

])

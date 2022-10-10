AC_DEFUN([AX_ENABLE_SHARED], [
  AM_CONDITIONAL([ENABLE_SHARED], [test "$enable_shared" = yes])
  AC_MSG_RESULT([$enable_shared])
])

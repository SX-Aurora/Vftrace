AC_DEFUN([AX_CHECK_FINSTRUMENT_OPTION], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   AX_CHECK_COMPILE_FLAG([-finstrument-functions],
      [AC_SUBST([FINSTRUMENT_FLAG],[-finstrument-functions])],
      [AX_CHECK_COMPILE_FLAG([-Xcompiler -finstrument-functions],
          [AC_SUBST([FINSTRUMENT_FLAG],[-Xcompiler -finstrument-functions])],
          [AX_CHECK_COMPILE_FLAG([-Minstrument=functions],
             [AC_SUBST([FINSTRUMENT_FLAG],[-Minstrument=functions])],
             [AC_MSG_ERROR([C compiler does not support any instrumentation option.])])])])])

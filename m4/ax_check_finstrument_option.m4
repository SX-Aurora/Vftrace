AC_DEFUN([AX_CHECK_FINSTRUMENT_OPTION_C], [
   AC_LANG(C)
   AC_PREREQ(2.50)
   [has_c_finstr=yes]
   AX_CHECK_COMPILE_FLAG([-finstrument-functions],
      [AC_SUBST([FINSTRUMENT_FLAG],[-finstrument-functions])],
      [AX_CHECK_COMPILE_FLAG([-Xcompiler -finstrument-functions],
          [AC_SUBST([FINSTRUMENT_FLAG],[-Xcompiler -finstrument-functions])],
          [AX_CHECK_COMPILE_FLAG([-Minstrument=functions],
             [AC_SUBST([FINSTRUMENT_FLAG],[-Minstrument=functions])],
             [has_c_finstr=no])])])
   AC_MSG_RESULT([$has_c_finstr])
   AM_CONDITIONAL([HAS_C_FINSTR], [test "x$has_c_finstr" = "xyes"])
])

AC_DEFUN([AX_CHECK_FINSTRUMENT_OPTION_FORTRAN], [
   AC_LANG(Fortran)
   AC_PREREQ(2.50)
   [has_f_instr=yes]
   AX_CHECK_COMPILE_FLAG([-finstrument-functions],
      [AC_SUBST([FINSTRUMENT_FLAG],[-finstrument-functions])],
      [AX_CHECK_COMPILE_FLAG([-Xcompiler -finstrument-functions],
          [AC_SUBST([FINSTRUMENT_FLAG],[-Xcompiler -finstrument-functions])],
          [AX_CHECK_COMPILE_FLAG([-Minstrument=functions],
             [AC_SUBST([FINSTRUMENT_FLAG],[-Minstrument=functions])],
             [has_f_finstr=no])])])
   AC_MSG_RESULT([$has_f_finstr])
   AM_CONDITIONAL([HAS_F_FINSTR], [test "x$has_f_finstr" = "xyes"])
])

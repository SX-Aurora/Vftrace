AC_DEFUN([AX_CHECK_INTEL_COMPILER], [
   AC_PREREQ(2.50)
   AC_LANG(C)
      AC_MSG_CHECKING([whether the Intel compiler is used])
      AC_RUN_IFELSE(
        [AC_LANG_SOURCE([[
int main() {
#ifdef __INTEL_LLVM_COMPILER
   return 0;
#else
   return 1;
#endif
}
        ]])],
         [uses_intel_compiler=yes],
         [uses_intel_compiler=no])
   AM_CONDITIONAL([USES_INTEL_COMPILER], [test "$uses_intel_compiler" = "yes"])
   AC_MSG_RESULT([$uses_intel_compiler])
   
])

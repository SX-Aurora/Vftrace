# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_LINK_USERFUNCTION_TEST_FILES
#
# DESCRIPTION
#
#   This macro links the test files and scripts to the test folders
#

AC_DEFUN([AX_LINK_USERFUNCTION_TEST_FILES], [
   AC_PREREQ(2.50)
   # User function tests
AC_CONFIG_LINKS(test/user_functions/cregions1.sh:test/user_functions/cregions1.sh
                test/user_functions/cregions2.sh:test/user_functions/cregions2.sh
                test/user_functions/cregions3.sh:test/user_functions/cregions3.sh
                test/user_functions/cregions4.sh:test/user_functions/cregions4.sh
                test/user_functions/fregions1.sh:test/user_functions/fregions1.sh
                test/user_functions/fregions2.sh:test/user_functions/fregions2.sh
                test/user_functions/fregions3.sh:test/user_functions/fregions3.sh
                test/user_functions/fregions4.sh:test/user_functions/fregions4.sh
               )
AC_CONFIG_LINKS(test/user_functions/cget_stack.sh:test/user_functions/cget_stack.sh
                test/user_functions/fget_stack.sh:test/user_functions/fget_stack.sh
               )
AC_CONFIG_LINKS(test/user_functions/cpause_resume.sh:test/user_functions/cpause_resume.sh
                test/user_functions/fpause_resume.sh:test/user_functions/fpause_resume.sh
               )
])

# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_LINK_TEST_FILES
#
# DESCRIPTION
#
#   This macro links the test files and scripts to the test folders
#

AC_DEFUN([AX_LINK_TEST_FILES], [
   AC_PREREQ(2.50)
   AX_LINK_UNIT_TEST_FILES
   AX_LINK_USERFUNCTION_TEST_FILES
   AX_LINK_CMPI_TEST_FILES
   AX_LINK_FMPI_TEST_FILES
   AX_LINK_F08MPI_TEST_FILES
])

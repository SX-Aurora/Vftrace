# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_MPI_STD
#
# DESCRIPTION
#
#   This macro checks which MPI standard is supported
#

AC_DEFUN([AX_CHECK_MPI_STD], [
   AC_PREREQ(2.50)
   mpi_version=none
   AM_COND_IF([ENABLE_MPI],[
      AC_MSG_CHECKING([which MPI-standard])
      AC_LANG(C)
      for maj_ver in 1 2 3 4;
      do
         for min_ver in 1 2 3;
         do
            AC_RUN_IFELSE([
               AC_LANG_PROGRAM(
               [[#include <mpi.h>]],
               [[return !(MPI_VERSION == ${maj_ver} && MPI_SUBVERSION == ${min_ver});]])
            ],[
               mpi_version=${maj_ver}.${min_ver}
               break 2
            ],[
               mpi_version=none
            ])
         done
      done
      AC_MSG_RESULT([$mpi_version])])
   AM_CONDITIONAL([VALID_MPI_VERSION],
                  [test "x$mpi_version" != "xnone"])
   AM_COND_IF([ENABLE_MPI],[
      AM_COND_IF([VALID_MPI_VERSION],
                 [],
                 [AC_MSG_ERROR([Unable to determine supported MPI-standard!])])])

   AM_COND_IF([ENABLE_MPI],[AX_COMPARE_VERSION([$mpi_version],[ge],[1.0],
      [has_mpi_std_1_0=yes],
      [has_mpi_std_1_0=no])])
   AM_CONDITIONAL([HAS_MPI_STD_1_0], [test "x$has_mpi_std_1_0" = "xyes"])

   AM_COND_IF([ENABLE_MPI],[AX_COMPARE_VERSION([$mpi_version],[ge],[1.1],
      [has_mpi_std_1_1=yes],
      [has_mpi_std_1_1=no])])
   AM_CONDITIONAL([HAS_MPI_STD_1_1], [test "x$has_mpi_std_1_1" = "xyes"])

   AM_COND_IF([ENABLE_MPI],[AX_COMPARE_VERSION([$mpi_version],[ge],[1.2],
      [has_mpi_std_1_2=yes],
      [has_mpi_std_1_2=no])])
   AM_CONDITIONAL([HAS_MPI_STD_1_2], [test "x$has_mpi_std_1_2" = "xyes"])

   AM_COND_IF([ENABLE_MPI],[AX_COMPARE_VERSION([$mpi_version],[ge],[1.3],
      [has_mpi_std_1_3=yes],
      [has_mpi_std_1_3=no])])
   AM_CONDITIONAL([HAS_MPI_STD_1_3], [test "x$has_mpi_std_1_3" = "xyes"])

   AM_COND_IF([ENABLE_MPI],[AX_COMPARE_VERSION([$mpi_version],[ge],[2.0],
      [has_mpi_std_2_0=yes],
      [has_mpi_std_2_0=no])])
   AM_CONDITIONAL([HAS_MPI_STD_2_0], [test "x$has_mpi_std_2_0" = "xyes"])

   AM_COND_IF([ENABLE_MPI],[AX_COMPARE_VERSION([$mpi_version],[ge],[2.1],
      [has_mpi_std_2_1=yes],
      [has_mpi_std_2_1=no])])
   AM_CONDITIONAL([HAS_MPI_STD_2_1], [test "x$has_mpi_std_2_1" = "xyes"])

   AM_COND_IF([ENABLE_MPI],[AX_COMPARE_VERSION([$mpi_version],[ge],[2.2],
      [has_mpi_std_2_2=yes],
      [has_mpi_std_2_2=no])])
   AM_CONDITIONAL([HAS_MPI_STD_2_2], [test "x$has_mpi_std_2_2" = "xyes"])

   AM_COND_IF([ENABLE_MPI],[AX_COMPARE_VERSION([$mpi_version],[ge],[3.0],
      [has_mpi_std_3_0=yes],
      [has_mpi_std_3_0=no])])
   AM_CONDITIONAL([HAS_MPI_STD_3_0], [test "x$has_mpi_std_3_0" = "xyes"])

   AM_COND_IF([ENABLE_MPI],[AX_COMPARE_VERSION([$mpi_version],[ge],[3.1],
      [has_mpi_std_3_1=yes],
      [has_mpi_std_3_1=no])])
   AM_CONDITIONAL([HAS_MPI_STD_3_1], [test "x$has_mpi_std_1_1" = "xyes"])

   AM_COND_IF([ENABLE_MPI],[AX_COMPARE_VERSION([$mpi_version],[ge],[4.0],
      [has_mpi_std_4_0=yes],
      [has_mpi_std_4_0=no])])
   AM_CONDITIONAL([HAS_MPI_STD_4_0], [test "x$has_mpi_std_4_0" = "xyes"])

   #AM_COND_IF([HAS_MPI_STD_1_0],[AC_MSG_NOTICE([MPI-1.0 = yes])], [AC_MSG_NOTICE([MPI-1.0 = no])])
   #AM_COND_IF([HAS_MPI_STD_1_1],[AC_MSG_NOTICE([MPI-1.1 = yes])], [AC_MSG_NOTICE([MPI-1.1 = no])])
   #AM_COND_IF([HAS_MPI_STD_1_2],[AC_MSG_NOTICE([MPI-1.2 = yes])], [AC_MSG_NOTICE([MPI-1.2 = no])])
   #AM_COND_IF([HAS_MPI_STD_1_3],[AC_MSG_NOTICE([MPI-1.3 = yes])], [AC_MSG_NOTICE([MPI-1.3 = no])])
   #AM_COND_IF([HAS_MPI_STD_2_0],[AC_MSG_NOTICE([MPI-2.0 = yes])], [AC_MSG_NOTICE([MPI-2.0 = no])])
   #AM_COND_IF([HAS_MPI_STD_2_1],[AC_MSG_NOTICE([MPI-2.1 = yes])], [AC_MSG_NOTICE([MPI-2.1 = no])])
   #AM_COND_IF([HAS_MPI_STD_2_2],[AC_MSG_NOTICE([MPI-2.2 = yes])], [AC_MSG_NOTICE([MPI-2.2 = no])])
   #AM_COND_IF([HAS_MPI_STD_3_0],[AC_MSG_NOTICE([MPI-3.0 = yes])], [AC_MSG_NOTICE([MPI-3.0 = no])])
   #AM_COND_IF([HAS_MPI_STD_3_1],[AC_MSG_NOTICE([MPI-3.1 = yes])], [AC_MSG_NOTICE([MPI-3.1 = no])])
   #AM_COND_IF([HAS_MPI_STD_4_0],[AC_MSG_NOTICE([MPI-4.0 = yes])], [AC_MSG_NOTICE([MPI-4.0 = no])])
])

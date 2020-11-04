# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_HWC_CONSISTENCY
#
# DESCRIPTION
#
#   This macro checks if HWC, architecture, and papi are enabled in a consistent way.
#

AC_DEFUN([AX_CHECK_HWC_CONSISTENCY], [
   AC_PREREQ(2.50)
   dnl # possible valid combinations are (X = off, O = on, - = does not matter)
   dnl # HWC  | X | O | O |
   dnl # SX   | - | O | X |
   dnl # Papi | X | X | O |
   dnl # every other combination is faulty
   AM_COND_IF(
      [ENABLE_HWC], [
         AM_COND_IF(
            [ON_VECTOR_ENGINE], [
               AM_COND_IF(
                  [HAS_PAPI], [
                     dnl# HWC + SX + Papi
                     AC_MSG_FAILURE([Hardware counters on a vector engine do not require Papi.])
                  ]
               )
            ], [
               AM_COND_IF(
                  [HAS_PAPI],,
                  [
                     dnl# HWC - SX - Papi
                     AC_MSG_FAILURE([Hardware counters require Papi.])
                  ]
               )
            ]
         )
      ],[
         AM_COND_IF(
            [HAS_PAPI], [
               dnl# -HWC + Papi
               AC_MSG_FAILURE([Without hardware counters papi is not required.])
            ]
         )
      ]
   )
])

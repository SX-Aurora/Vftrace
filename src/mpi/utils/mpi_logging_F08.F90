MODULE vftr_mpi_logging_F08

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_no_mpi_logging_F08

   INTERFACE

      FUNCTION vftr_no_mpi_logging_int_F08() &
         BIND(c, NAME="vftr_no_mpi_logging_int")
         USE ISO_C_BINDING, ONLY : C_INT
         IMPLICIT NONE
         INTEGER(KIND=C_INT) :: vftr_no_mpi_logging_int_F08
      END FUNCTION vftr_no_mpi_logging_int_F08

   END INTERFACE

CONTAINS

   LOGICAL FUNCTION vftr_no_mpi_logging_F08()
      vftr_no_mpi_logging_F08 = vftr_no_mpi_logging_int_F08() == 1
      RETURN
   END FUNCTION vftr_no_mpi_logging_F08

END MODULE vftr_mpi_logging_F08

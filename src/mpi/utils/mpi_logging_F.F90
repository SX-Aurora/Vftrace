MODULE vftr_mpi_logging_F

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_no_mpi_logging_F

   INTERFACE

      FUNCTION vftr_no_mpi_logging_int_F() &
         BIND(c, NAME="vftr_no_mpi_logging_int")
         USE ISO_C_BINDING, ONLY : C_INT
         IMPLICIT NONE
         INTEGER(KIND=C_INT) :: vftr_no_mpi_logging_int_F
      END FUNCTION vftr_no_mpi_logging_int_F

   END INTERFACE

CONTAINS

   LOGICAL FUNCTION vftr_no_mpi_logging_F()
      vftr_no_mpi_logging_F = vftr_no_mpi_logging_int_F() == 1
      RETURN
   END FUNCTION vftr_no_mpi_logging_F

END MODULE vftr_mpi_logging_F

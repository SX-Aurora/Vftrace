MODULE vftr_mpi_logging_f08
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_no_mpi_logging_f08

   INTERFACE

      FUNCTION vftr_no_mpi_logging_int_f08() &
         BIND(c, NAME="vftr_no_mpi_logging_int")
         USE ISO_C_BINDING, ONLY : C_INT
         IMPLICIT NONE
         INTEGER(KIND=C_INT) :: vftr_no_mpi_logging_int_f08
      END FUNCTION vftr_no_mpi_logging_int_f08

   END INTERFACE

#endif

CONTAINS

#ifdef _MPI
   LOGICAL FUNCTION vftr_no_mpi_logging_f08()
      vftr_no_mpi_logging_f08 = vftr_no_mpi_logging_int_f08() == 1
      RETURN
   END FUNCTION vftr_no_mpi_logging_f08
#endif

END MODULE vftr_mpi_logging_f08

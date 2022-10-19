MODULE vftr_mpi_finalize_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_mpi_finalize_f2vftr

   INTERFACE

      SUBROUTINE vftr_mpi_finalize_f2vftr(IERROR) &
         BIND(c, NAME="vftr_MPI_Finalize_f2vftr")
         IMPLICIT NONE
         INTEGER :: IERROR
      END SUBROUTINE vftr_mpi_finalize_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_finalize_f2vftr_fi

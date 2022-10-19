MODULE vftr_mpi_wait_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Wait_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Wait_f2vftr(F_REQUEST, F_STATUS, F_ERROR) &
         BIND(C, name="vftr_MPI_Wait_f2vftr")
         USE mpi, ONLY: MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER F_REQUEST
         INTEGER F_STATUS(MPI_STATUS_SIZE)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Wait_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_wait_f2vftr_fi

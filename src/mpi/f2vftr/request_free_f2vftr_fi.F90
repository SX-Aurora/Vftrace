MODULE vftr_mpi_request_free_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Request_free_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Request_free_f2vftr(F_REQUEST, F_ERROR) &
         BIND(C, name="vftr_MPI_Request_free_f2vftr")
         IMPLICIT NONE
         INTEGER F_REQUEST
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Request_free_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_request_free_f2vftr_fi

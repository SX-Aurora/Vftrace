MODULE vftr_mpi_request_free_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Request_free_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Request_free_f082vftr(f_request, f_error) &
         BIND(C, name="vftr_MPI_Request_free_f082vftr")
         IMPLICIT NONE
         INTEGER, INTENT(INOUT) :: F_REQUEST
         INTEGER, INTENT(OUT) :: F_ERROR
      END SUBROUTINE vftr_MPI_Request_free_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_request_free_f082vftr_f08i

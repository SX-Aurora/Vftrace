MODULE vftr_mpi_igatherv_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Igatherv_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Igatherv_f082vftr(sendbuf, sendcount, f_sendtype, &
                                           recvbuf, f_recvcounts, f_displs, &
                                           f_recvtype, root, f_comm, &
                                           f_request, f_error) &
         BIND(C, name="vftr_MPI_Igatherv_f082vftr")
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: sendbuf
         INTEGER, INTENT(IN) :: sendcount
         INTEGER, INTENT(IN) :: f_sendtype
         INTEGER, INTENT(IN) :: recvbuf
         INTEGER, INTENT(IN) :: f_recvcounts(*)
         INTEGER, INTENT(IN) :: f_displs(*)
         INTEGER, INTENT(IN) :: f_recvtype
         INTEGER, INTENT(IN) :: root
         INTEGER, INTENT(IN) :: f_comm
         INTEGER, INTENT(IN) :: f_request
         INTEGER, INTENT(OUT) :: f_error
      END SUBROUTINE vftr_MPI_Igatherv_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_igatherv_f082vftr_f08i

MODULE vftr_mpi_gatherv_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Gatherv_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Gatherv_f082vftr(sendbuf, sendcount, f_sendtype, &
                                           recvbuf, f_recvcounts, f_displs, &
                                           f_recvtype, root, f_comm, &
                                           f_error) &
         BIND(C, name="vftr_MPI_Gatherv_f082vftr")
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: SENDBUF
         INTEGER, INTENT(IN) :: SENDCOUNT
         INTEGER, INTENT(IN) :: F_SENDTYPE
         INTEGER, INTENT(IN) :: RECVBUF
         INTEGER, INTENT(IN) :: F_RECVCOUNTS(*)
         INTEGER, INTENT(IN) :: F_DISPLS(*)
         INTEGER, INTENT(IN) :: F_RECVTYPE
         INTEGER, INTENT(IN) :: ROOT
         INTEGER, INTENT(IN) :: F_COMM
         INTEGER, INTENT(OUT) :: F_ERROR
      END SUBROUTINE vftr_MPI_Gatherv_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_gatherv_f082vftr_f08i

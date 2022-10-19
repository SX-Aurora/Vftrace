MODULE vftr_mpi_alltoallw_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Alltoallw_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Alltoallw_f082vftr(sendbuf, f_sendcounts, f_sdispls, f_sendtypes, &
                                             recvbuf, f_recvcounts, f_rdispls, f_recvtypes, &
                                             f_comm, f_error) &
         BIND(C, name="vftr_MPI_Alltoallw_f082vftr")
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: sendbuf
         INTEGER, INTENT(IN) :: f_sendcounts(*)
         INTEGER, INTENT(IN) :: f_sdispls(*)
         INTEGER, INTENT(IN) :: f_sendtypes(*)
         INTEGER, INTENT(IN) :: recvbuf
         INTEGER, INTENT(IN) :: f_recvcounts(*)
         INTEGER, INTENT(IN) :: f_rdispls(*)
         INTEGER, INTENT(IN) :: f_recvtypes(*)
         INTEGER, INTENT(IN) :: f_comm
         INTEGER, INTENT(OUT) :: f_error
      END SUBROUTINE vftr_MPI_Alltoallw_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_alltoallw_f082vftr_f08i

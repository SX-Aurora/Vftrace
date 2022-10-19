MODULE vftr_mpi_neighbor_alltoall_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Neighbor_alltoall_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Neighbor_alltoall_f082vftr(sendbuf, sendcount, f_sendtype, &
                                                     recvbuf, recvcount, f_recvtype, &
                                                     f_comm, f_error) &
         BIND(C, name="vftr_MPI_Neighbor_alltoall_f082vftr")
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: sendbuf
         INTEGER, INTENT(IN) :: sendcount
         INTEGER, INTENT(IN) :: f_sendtype
         INTEGER, INTENT(IN) :: recvbuf
         INTEGER, INTENT(IN) :: recvcount
         INTEGER, INTENT(IN) :: f_recvtype
         INTEGER, INTENT(IN) :: f_comm
         INTEGER, INTENT(OUT) :: f_error
      END SUBROUTINE vftr_MPI_Neighbor_alltoall_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_neighbor_alltoall_f082vftr_f08i

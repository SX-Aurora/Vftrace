MODULE vftr_mpi_reduce_scatter_block_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Reduce_scatter_block_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Reduce_scatter_block_f082vftr(sendbuf, recvbuf, recvcount, &
                                                        f_datatype, f_op, f_comm, &
                                                        f_error) &
         BIND(C, name="vftr_MPI_Reduce_scatter_block_f082vftr")
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: sendbuf
         INTEGER, INTENT(IN) :: recvbuf
         INTEGER, INTENT(IN) :: recvcount
         INTEGER, INTENT(IN) :: f_datatype
         INTEGER, INTENT(IN) :: f_op
         INTEGER, INTENT(IN) :: f_comm
         INTEGER, INTENT(OUT) :: f_error
      END SUBROUTINE vftr_MPI_Reduce_scatter_block_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_reduce_scatter_block_f082vftr_f08i

MODULE vftr_mpi_sendrecv_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Sendrecv_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Sendrecv_f082vftr(sendbuf, sendcount, f_sendtype, dest, sendtag, &
                                            recvbuf, recvcount, f_recvtype, source, recvtag, &
                                            f_comm, f_status, f_error) &
         BIND(C, name="vftr_MPI_Sendrecv_f082vftr")
         USE mpi_f08, &
            ONLY : MPI_Status
         IMPLICIT NONE
         INTEGER :: sendbuf
         INTEGER :: sendcount
         INTEGER :: f_sendtype
         INTEGER :: dest
         INTEGER :: sendtag
         INTEGER :: recvbuf
         INTEGER :: recvcount
         INTEGER :: f_recvtype
         INTEGER :: source
         INTEGER :: recvtag
         INTEGER :: f_comm
         TYPE(MPI_Status) :: f_status
         INTEGER :: f_error
      END SUBROUTINE vftr_MPI_Sendrecv_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_sendrecv_f082vftr_f08i

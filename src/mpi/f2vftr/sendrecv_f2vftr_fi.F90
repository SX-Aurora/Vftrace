MODULE vftr_mpi_sendrecv_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Sendrecv_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Sendrecv_f2vftr(SENDBUF, SENDCOUNT, F_SENDTYPE, DEST, SENDTAG, &
                                       RECVBUF, RECVCOUNT, F_RECVTYPE, SOURCE, RECVTAG, &
                                       F_COMM, F_STATUS, F_ERROR) &
         BIND(C, name="vftr_MPI_Sendrecv_f2vftr")
         USE mpi, ONLY : MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER SENDCOUNT
         INTEGER F_SENDTYPE
         INTEGER DEST
         INTEGER SENDTAG
         INTEGER RECVBUF
         INTEGER RECVCOUNT
         INTEGER F_RECVTYPE
         INTEGER SOURCE
         INTEGER RECVTAG
         INTEGER F_COMM
         INTEGER F_STATUS(MPI_STATUS_SIZE)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Sendrecv_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_sendrecv_f2vftr_fi

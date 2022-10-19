MODULE vftr_mpi_sendrecv_replace_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Sendrecv_replace_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Sendrecv_replace_f2vftr(BUF, COUNT, F_DATATYPE, DEST, SENDTAG, SOURCE, &
                                               RECVTAG, F_COMM, F_STATUS, F_ERROR) &
         BIND(C, name="vftr_MPI_Sendrecv_replace_f2vftr")
         USE mpi, ONLY : MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER DEST
         INTEGER SENDTAG
         INTEGER SOURCE
         INTEGER RECVTAG
         INTEGER F_COMM
         INTEGER F_STATUS(MPI_STATUS_SIZE)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Sendrecv_replace_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_sendrecv_replace_f2vftr_fi

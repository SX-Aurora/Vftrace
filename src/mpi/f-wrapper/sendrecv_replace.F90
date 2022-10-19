#ifdef _MPI

SUBROUTINE MPI_SENDRECV_REPLACE(BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, &
                                RECVTAG, COMM, STATUS, ERROR)
   USE vftr_mpi_sendrecv_replace_f2vftr_fi, &
      ONLY : vftr_MPI_Sendrecv_replace_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_SENDRECV_REPLACE, &
             MPI_STATUS_SIZE
   IMPLICIT NONE
   INTEGER BUF
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER DEST
   INTEGER SENDTAG
   INTEGER SOURCE
   INTEGER RECVTAG
   INTEGER COMM
   INTEGER STATUS(MPI_STATUS_SIZE)
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_SENDRECV_REPLACE(BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, &
                                 RECVTAG, COMM, STATUS, ERROR)
   ELSE
      CALL vftr_MPI_Sendrecv_replace_f2vftr(BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, &
                                         RECVTAG, COMM, STATUS, ERROR)
   END IF

END SUBROUTINE MPI_SENDRECV_REPLACE

#endif

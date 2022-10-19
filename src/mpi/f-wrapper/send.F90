#ifdef _MPI

SUBROUTINE MPI_SEND(BUF, COUNT, DATATYPE, DEST, TAG, COMM, ERROR)
   USE vftr_mpi_send_f2vftr_fi, &
      ONLY : vftr_MPI_Send_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_SEND
   IMPLICIT NONE
   INTEGER BUF
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER DEST
   INTEGER TAG
   INTEGER COMM
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_SEND(BUF, COUNT, DATATYPE, DEST, TAG, COMM, ERROR)
   ELSE
      CALL vftr_MPI_Send_f2vftr(BUF, COUNT, DATATYPE, DEST, TAG, COMM, ERROR)
   END IF

END SUBROUTINE MPI_SEND

#endif

#ifdef _MPI

SUBROUTINE MPI_BSEND(BUF, COUNT, DATATYPE, DEST, TAG, COMM, ERROR)
   USE vftr_mpi_bsend_f2vftr_fi, &
      ONLY : vftr_MPI_Bsend_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_BSEND
   IMPLICIT NONE
   INTEGER BUF
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER DEST
   INTEGER TAG
   INTEGER COMM
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_BSEND(BUF, COUNT, DATATYPE, DEST, TAG, COMM, ERROR)
   ELSE
      CALL vftr_MPI_Bsend_f2vftr(BUF, COUNT, DATATYPE, DEST, TAG, COMM, ERROR)
   END IF

END SUBROUTINE MPI_BSEND

#endif

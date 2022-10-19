#ifdef _MPI

SUBROUTINE MPI_RSEND(BUF, COUNT, DATATYPE, DEST, TAG, COMM, ERROR)
   USE vftr_mpi_rsend_f2vftr_fi, &
      ONLY : vftr_MPI_Rsend_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_RSEND
   IMPLICIT NONE
   INTEGER BUF
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER DEST
   INTEGER TAG
   INTEGER COMM
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_RSEND(BUF, COUNT, DATATYPE, DEST, TAG, COMM, ERROR)
   ELSE
      CALL vftr_MPI_Rsend_f2vftr(BUF, COUNT, DATATYPE, DEST, TAG, COMM, ERROR)
   END IF

END SUBROUTINE MPI_RSEND

#endif

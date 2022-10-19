#ifdef _MPI

SUBROUTINE MPI_IBARRIER(COMM, REQUEST, ERROR)
   USE vftr_mpi_ibarrier_f2vftr_fi, &
      ONLY : vftr_MPI_Ibarrier_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_BARRIER
   IMPLICIT NONE
   INTEGER COMM
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IBARRIER(COMM, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Ibarrier_f2vftr(COMM, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IbARRIER

#endif

#ifdef _MPI

SUBROUTINE MPI_WAIT(REQUEST, STATUS, ERROR)
   USE vftr_mpi_wait_f2vftr_fi, &
      ONLY : vftr_MPI_Wait_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY: PMPI_WAIT, &
            MPI_STATUS_SIZE
   IMPLICIT NONE
   INTEGER REQUEST
   INTEGER STATUS(MPI_STATUS_SIZE)
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_WAIT(REQUEST, STATUS, ERROR)
   ELSE
      CALL vftr_MPI_Wait_f2vftr(REQUEST, STATUS, ERROR)
   END IF

END SUBROUTINE MPI_WAIT

#endif

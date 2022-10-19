#ifdef _MPI

SUBROUTINE MPI_BARRIER(COMM, ERROR)
   USE vftr_mpi_barrier_f2vftr_fi, &
      ONLY : vftr_MPI_Barrier_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_BARRIER
   IMPLICIT NONE
   INTEGER COMM
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_BARRIER(COMM, ERROR)
   ELSE
      CALL vftr_MPI_Barrier_f2vftr(COMM, ERROR)
   END IF

END SUBROUTINE MPI_BARRIER

#endif

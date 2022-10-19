#ifdef _MPI

SUBROUTINE MPI_Barrier_f08(comm, error)
   USE vftr_mpi_barrier_f082vftr_f08i, &
      ONLY : vftr_MPI_Barrier_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Barrier, &
             MPI_Comm
   IMPLICIT NONE
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Barrier(comm, tmperror)
   ELSE
      CALL vftr_MPI_Barrier_f082vftr(comm%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Barrier_f08

#endif

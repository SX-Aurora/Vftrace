#ifdef _MPI

SUBROUTINE MPI_Probe_f08(source, tag, comm, status, error)
   USE vftr_mpi_probe_f082vftr_f08i, &
      ONLY : vftr_MPI_Probe_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Probe, &
             MPI_Comm, &
             MPI_Status
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: source
   INTEGER, INTENT(IN) :: tag
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Status) :: status
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Probe(source, tag, comm, status, tmperror)
   ELSE
      CALL vftr_MPI_Probe_f082vftr(source, tag, comm%MPI_VAL, status, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Probe_f08

#endif

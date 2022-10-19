#ifdef _MPI

SUBROUTINE MPI_PROBE(SOURCE, TAG, COMM, STATUS, ERROR)
   USE vftr_mpi_probe_f2vftr_fi, &
      ONLY : vftr_MPI_Probe_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_PROBE, &
             MPI_STATUS_SIZE
   IMPLICIT NONE
   INTEGER SOURCE
   INTEGER TAG
   INTEGER COMM
   INTEGER STATUS(MPI_STATUS_SIZE)
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_PROBE(SOURCE, TAG, COMM, STATUS, ERROR)
   ELSE
      CALL vftr_MPI_Probe_f2vftr(SOURCE, TAG, COMM, STATUS, ERROR)
   END IF

END SUBROUTINE MPI_PROBE

#endif

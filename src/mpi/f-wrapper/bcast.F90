#ifdef _MPI

SUBROUTINE MPI_BCAST(BUFFER, COUNT, DATATYPE, &
                     ROOT, COMM, ERROR)
   USE vftr_mpi_bcast_f2vftr_fi, &
      ONLY : vftr_MPI_Bcast_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_BCAST
   IMPLICIT NONE
   INTEGER BUFFER
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER ROOT
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_estimate_sync_time("MPI_Bcast_sync", COMM);

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_BCAST(BUFFER, COUNT, DATATYPE, &
                      ROOT, COMM, ERROR)
   ELSE
      CALL vftr_MPI_Bcast_f2vftr(BUFFER, COUNT, DATATYPE, &
                              ROOT, COMM, ERROR)
   END IF

END SUBROUTINE MPI_BCAST

#endif

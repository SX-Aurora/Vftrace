#ifdef _MPI

SUBROUTINE MPI_Bcast_f08(buffer, count, datatype, &
                         root, comm, error)
   USE vftr_mpi_bcast_f082vftr_f08i, &
      ONLY : vftr_MPI_Bcast_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE vftr_sync_time_F08, &
      ONLY : vftr_estimate_sync_time
   USE mpi_f08, &
      ONLY : PMPI_Bcast, &
             MPI_Datatype, &
             MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: buffer
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: root
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_estimate_sync_time("MPI_Bcast_sync", comm)

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Bcast(buffer, count, datatype, &
                      root, comm, tmperror)
   ELSE
      CALL vftr_MPI_Bcast_f082vftr(buffer, count, datatype%MPI_VAL, &
                                root, comm%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Bcast_f08

#endif

#ifdef _MPI

SUBROUTINE MPI_Scan_f08(sendbuf, recvbuf, count, &
                        datatype, op, comm, error)
   USE vftr_mpi_scan_f082vftr_f08i, &
      ONLY : vftr_MPI_Scan_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE vftr_sync_time_f08, &
      ONLY : vftr_estimate_sync_time
   USE mpi_f08, &
      ONLY : PMPI_Scan, &
             MPI_Datatype, &
             MPI_Op, &
             MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   TYPE(MPI_Op), INTENT(IN) :: op
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_estimate_sync_time("MPI_Scan_sync", comm)

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Scan(sendbuf, recvbuf, count, &
                     datatype, op, &
                     comm, tmperror)
   ELSE
      CALL vftr_MPI_Scan_f082vftr(sendbuf, recvbuf, count, &
                               datatype%MPI_VAL, op%MPI_VAL, &
                               comm%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Scan_f08

#endif

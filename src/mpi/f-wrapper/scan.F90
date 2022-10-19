#ifdef _MPI

SUBROUTINE MPI_SCAN(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                    OP, COMM, ERROR)
   USE vftr_mpi_scan_f2vftr_fi, &
      ONLY : vftr_MPI_Scan_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_SCAN
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER OP
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_estimate_sync_time("MPI_Scan_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_SCAN(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                     OP, COMM, ERROR)
   ELSE
      CALL vftr_MPI_Scan_f2vftr(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                             OP, COMM, ERROR)
   END IF

END SUBROUTINE MPI_SCAN

#endif

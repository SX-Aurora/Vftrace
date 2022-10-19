#ifdef _MPI

SUBROUTINE MPI_EXSCAN(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                      OP, COMM, ERROR)
   USE vftr_mpi_exscan_f2vftr_fi, &
      ONLY : vftr_MPI_Exscan_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_EXSCAN
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER OP
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_estimate_sync_time("MPI_Exscan_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_EXSCAN(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                       OP, COMM, ERROR)
   ELSE
      CALL vftr_MPI_Exscan_f2vftr(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                               OP, COMM, ERROR)
   END IF

END SUBROUTINE MPI_EXSCAN

#endif

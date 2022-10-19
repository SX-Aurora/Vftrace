#ifdef _MPI

SUBROUTINE MPI_REDUCE(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                      OP, ROOT, COMM, ERROR)
   USE vftr_mpi_reduce_f2vftr_fi, &
      ONLY : vftr_MPI_Reduce_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_REDUCE
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER OP
   INTEGER ROOT
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_estimate_sync_time("MPI_Reduce_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_REDUCE(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                       OP, ROOT, COMM, ERROR)
   ELSE
      CALL vftr_MPI_Reduce_f2vftr(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                               OP, ROOT, COMM, ERROR)
   END IF

END SUBROUTINE MPI_REDUCE

#endif

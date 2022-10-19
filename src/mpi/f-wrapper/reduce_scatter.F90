#ifdef _MPI

SUBROUTINE MPI_REDUCE_SCATTER(SENDBUF, RECVBUF, RECVCOUNTS, &
                              DATATYPE, OP, COMM, &
                              ERROR)
   USE vftr_mpi_reduce_scatter_f2vftr_fi, &
      ONLY : vftr_MPI_Reduce_scatter_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_REDUCE_SCATTER
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER RECVCOUNTS(*)
   INTEGER DATATYPE
   INTEGER OP
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_estimate_sync_time("MPI_Reduce_scatter_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_REDUCE_SCATTER(SENDBUF, RECVBUF, RECVCOUNTS, &
                               DATATYPE, OP, COMM, &
                               ERROR)
   ELSE
      CALL vftr_MPI_Reduce_scatter_f2vftr(SENDBUF, RECVBUF, RECVCOUNTS, &
                                       DATATYPE, OP, COMM, &
                                       ERROR)
   END IF

END SUBROUTINE MPI_REDUCE_SCATTER

#endif

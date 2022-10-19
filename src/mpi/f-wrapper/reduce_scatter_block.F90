#ifdef _MPI

SUBROUTINE MPI_REDUCE_SCATTER_BLOCK(SENDBUF, RECVBUF, RECVCOUNT, &
                                    DATATYPE, OP, COMM, &
                                    ERROR)
   USE vftr_mpi_reduce_scatter_block_f2vftr_fi, &
      ONLY : vftr_MPI_Reduce_scatter_block_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_REDUCE_SCATTER_BLOCK
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER RECVCOUNT
   INTEGER DATATYPE
   INTEGER OP
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_estimate_sync_time("MPI_Reduce_scatter_block_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_REDUCE_SCATTER_BLOCK(SENDBUF, RECVBUF, RECVCOUNT, &
                                     DATATYPE, OP, COMM, &
                                     ERROR)
   ELSE
      CALL vftr_MPI_Reduce_scatter_block_f2vftr(SENDBUF, RECVBUF, RECVCOUNT, &
                                             DATATYPE, OP, COMM, &
                                             ERROR)
   END IF

END SUBROUTINE MPI_REDUCE_SCATTER_BLOCK

#endif

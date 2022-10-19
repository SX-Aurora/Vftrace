#ifdef _MPI

SUBROUTINE MPI_IREDUCE_SCATTER_BLOCK(SENDBUF, RECVBUF, RECVCOUNT, &
                                     DATATYPE, OP, COMM, &
                                     REQUEST, ERROR)
   USE vftr_mpi_ireduce_scatter_block_f2vftr_fi, &
      ONLY : vftr_MPI_Ireduce_scatter_block_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IREDUCE_SCATTER_BLOCK
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER RECVCOUNT
   INTEGER DATATYPE
   INTEGER OP
   INTEGER COMM
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IREDUCE_SCATTER_BLOCK(SENDBUF, RECVBUF, RECVCOUNT, &
                                      DATATYPE, OP, COMM, &
                                      REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Ireduce_scatter_block_f2vftr(SENDBUF, RECVBUF, RECVCOUNT, &
                                              DATATYPE, OP, COMM, &
                                              REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IREDUCE_SCATTER_BLOCK

#endif

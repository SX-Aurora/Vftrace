#ifdef _MPI

SUBROUTINE MPI_IREDUCE_SCATTER(SENDBUF, RECVBUF, RECVCOUNTS, &
                               DATATYPE, OP, COMM, &
                               REQUEST, ERROR)
   USE vftr_mpi_ireduce_scatter_f2vftr_fi, &
      ONLY : vftr_MPI_Ireduce_scatter_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IREDUCE_SCATTER
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER RECVCOUNTS(*)
   INTEGER DATATYPE
   INTEGER OP
   INTEGER COMM
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IREDUCE_SCATTER(SENDBUF, RECVBUF, RECVCOUNTS, &
                                DATATYPE, OP, COMM, &
                                REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Ireduce_scatter_f2vftr(SENDBUF, RECVBUF, RECVCOUNTS, &
                                        DATATYPE, OP, COMM, &
                                        REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IREDUCE_SCATTER

#endif

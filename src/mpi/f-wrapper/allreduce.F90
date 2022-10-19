#ifdef _MPI

SUBROUTINE MPI_ALLREDUCE(SENDBUF, RECVBUF, COUNT, &
                         DATATYPE, OP, COMM, &
                         ERROR)
   USE vftr_mpi_allreduce_f2vftr_fi, &
      ONLY : vftr_MPI_Allreduce_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_ALLREDUCE
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER OP
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_estimate_sync_time("MPI_Allreduce_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_ALLREDUCE(SENDBUF, RECVBUF, COUNT, &
                          DATATYPE, OP, COMM, &
                          ERROR)
   ELSE
      CALL vftr_MPI_Allreduce_f2vftr(SENDBUF, RECVBUF, COUNT, &
                                  DATATYPE, OP, COMM, &
                                  ERROR)
   END IF

END SUBROUTINE MPI_ALLREDUCE

#endif

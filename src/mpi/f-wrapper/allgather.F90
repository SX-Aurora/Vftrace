#ifdef _MPI

SUBROUTINE MPI_ALLGATHER(SENDBUF, SENDCOUNT, SENDTYPE, &
                         RECVBUF, RECVCOUNT, RECVTYPE, &
                         COMM, ERROR)
   USE vftr_mpi_allgather_f2vftr_fi, &
      ONLY : vftr_MPI_Allgather_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_ALLGATHER
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNT
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNT
   INTEGER RECVTYPE
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_estimate_sync_time("MPI_Allgather_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_ALLGATHER(SENDBUF, SENDCOUNT, SENDTYPE, &
                          RECVBUF, RECVCOUNT, RECVTYPE, &
                          COMM, ERROR)
   ELSE
      CALL vftr_MPI_Allgather_f2vftr(SENDBUF, SENDCOUNT, SENDTYPE, &
                                  RECVBUF, RECVCOUNT, RECVTYPE, &
                                  COMM, ERROR)
   END IF

END SUBROUTINE MPI_ALLGATHER

#endif

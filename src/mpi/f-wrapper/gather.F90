#ifdef _MPI

SUBROUTINE MPI_GATHER(SENDBUF, SENDCOUNT, SENDTYPE, &
                      RECVBUF, RECVCOUNT, RECVTYPE, &
                      ROOT, COMM, ERROR)
   USE vftr_mpi_gather_f2vftr_fi, &
      ONLY : vftr_MPI_Gather_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_GATHER
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNT
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNT
   INTEGER RECVTYPE
   INTEGER ROOT
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_estimate_sync_time("MPI_Gather_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_GATHER(SENDBUF, SENDCOUNT, SENDTYPE, &
                       RECVBUF, RECVCOUNT, RECVTYPE, &
                       ROOT, COMM, ERROR)
   ELSE
      CALL vftr_MPI_Gather_f2vftr(SENDBUF, SENDCOUNT, SENDTYPE, &
                               RECVBUF, RECVCOUNT, RECVTYPE, &
                               ROOT, COMM, ERROR)
   END IF

END SUBROUTINE MPI_GATHER

#endif

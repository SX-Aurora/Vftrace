#ifdef _MPI

SUBROUTINE MPI_ALLTOALLV(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                         RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                         COMM, ERROR)
   USE vftr_mpi_alltoallv_f2vftr_fi, &
      ONLY : vftr_MPI_Alltoallv_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_ALLTOALLV
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNTS(*)
   INTEGER SDISPLS(*)
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNTS(*)
   INTEGER RDISPLS(*)
   INTEGER RECVTYPE
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_estimate_sync_time("MPI_Alltoallv_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_ALLTOALLV(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                          RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                          COMM, ERROR)
   ELSE
      CALL vftr_MPI_Alltoallv_f2vftr(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                                  RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                                  COMM, ERROR)
   END IF

END SUBROUTINE MPI_ALLTOALLV

#endif

#ifdef _MPI

SUBROUTINE MPI_NEIGHBOR_ALLTOALLV(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                                  RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                                  COMM, ERROR)
   USE vftr_mpi_neighbor_alltoallv_f2vftr_fi, &
      ONLY : vftr_MPI_Neighbor_alltoallv_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_NEIGHBOR_ALLTOALLV
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

   CALL vftr_estimate_sync_time("MPI_Neighbor_alltoallv_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_NEIGHBOR_ALLTOALLV(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                                   RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                                   COMM, ERROR)
   ELSE
      CALL vftr_MPI_Neighbor_alltoallv_f2vftr(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                                              RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                                              COMM, ERROR)
   END IF

END SUBROUTINE MPI_NEIGHBOR_ALLTOALLV
#endif

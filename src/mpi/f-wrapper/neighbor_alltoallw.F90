#ifdef _MPI

SUBROUTINE MPI_NEIGHBOR_ALLTOALLW(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPES, &
                                  RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPES, &
                                  COMM, ERROR)
   USE vftr_mpi_neighbor_alltoallw_f2vftr_fi, &
      ONLY : vftr_MPI_Neighbor_alltoallw_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : MPI_ADDRESS_KIND, &
             PMPI_NEIGHBOR_ALLTOALLW
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNTS(*)
   INTEGER(KIND=MPI_ADDRESS_KIND) SDISPLS(*)
   INTEGER SENDTYPES(*)
   INTEGER RECVBUF
   INTEGER RECVCOUNTS(*)
   INTEGER(KIND=MPI_ADDRESS_KIND) RDISPLS(*)
   INTEGER RECVTYPES(*)
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_estimate_sync_time("MPI_Neighbor_alltoallw_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_NEIGHBOR_ALLTOALLW(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPES, &
                                   RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPES, &
                                   COMM, ERROR)
   ELSE
      CALL vftr_MPI_Neighbor_alltoallw_f2vftr(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPES, &
                                              RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPES, &
                                              COMM, ERROR)
  END IF

END SUBROUTINE MPI_NEIGHBOR_ALLTOALLW

#endif

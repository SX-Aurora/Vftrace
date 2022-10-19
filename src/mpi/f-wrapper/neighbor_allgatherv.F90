#ifdef _MPI

SUBROUTINE MPI_NEIGHBOR_ALLGATHERV(SENDBUF, SENDCOUNT, SENDTYPE, &
                                   RECVBUF, RECVCOUNTS, DISPLS, &
                                   RECVTYPE, COMM, ERROR)
   USE vftr_mpi_neighbor_allgatherv_f2vftr_fi, &
      ONLY : vftr_MPI_Neighbor_allgatherv_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_NEIGHBOR_ALLGATHERV
   IMPLICIT NONE
   INTEGER ::  SENDBUF
   INTEGER ::  SENDCOUNT
   INTEGER ::  SENDTYPE
   INTEGER ::  RECVBUF
   INTEGER ::  RECVCOUNTS(*)
   INTEGER ::  DISPLS(*)
   INTEGER ::  RECVTYPE
   INTEGER ::  COMM
   INTEGER ::  ERROR

   CALL vftr_estimate_sync_time("MPI_Neighbor_allgatherv_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_NEIGHBOR_ALLGATHERV(SENDBUF, SENDCOUNT, SENDTYPE, &
                                    RECVBUF, RECVCOUNTS, DISPLS, &
                                    RECVTYPE, COMM, ERROR)
   ELSE
      CALL vftr_MPI_Neighbor_allgatherv_f2vftr(SENDBUF, SENDCOUNT, SENDTYPE, &
                                               RECVBUF, RECVCOUNTS, DISPLS, &
                                               RECVTYPE, COMM, ERROR)
   END IF

END SUBROUTINE MPI_NEIGHBOR_ALLGATHERV

#endif

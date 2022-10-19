#ifdef _MPI

SUBROUTINE MPI_INEIGHBOR_ALLTOALLV(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                                   RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                                   COMM, REQUEST, ERROR)
   USE vftr_mpi_ineighbor_alltoallv_f2vftr_fi, &
      ONLY : vftr_MPI_Ineighbor_alltoallv_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
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
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_INEIGHBOR_ALLTOALLV(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                                    RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                                    COMM, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Ineighbor_alltoallv_f2vftr(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                                               RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                                               COMM, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_INEIGHBOR_ALLTOALLV

#endif

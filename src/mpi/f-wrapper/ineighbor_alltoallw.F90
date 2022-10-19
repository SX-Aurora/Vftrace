#ifdef _MPI

SUBROUTINE MPI_INEIGHBOR_ALLTOALLW(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPES, &
                                   RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPES, &
                                   COMM, REQUEST, ERROR)
   USE vftr_mpi_ineighbor_alltoallw_f2vftr_fi, &
      ONLY : vftr_MPI_Ineighbor_alltoallw_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : MPI_ADDRESS_KIND, &
             PMPI_INEIGHBOR_ALLTOALLW
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
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_INEIGHBOR_ALLTOALLW(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPES, &
                                    RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPES, &
                                    COMM, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Ineighbor_alltoallw_f2vftr(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPES, &
                                               RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPES, &
                                               COMM, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_INEIGHBOR_ALLTOALLW

#endif

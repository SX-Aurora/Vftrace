#ifdef _MPI

SUBROUTINE MPI_IALLTOALLW(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPES, &
                          RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPES, &
                          COMM, REQUEST, ERROR)
   USE vftr_mpi_ialltoallw_f2vftr_fi, &
      ONLY : vftr_MPI_Ialltoallw_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IALLTOALLW
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNTS(*)
   INTEGER SDISPLS(*)
   INTEGER SENDTYPES(*)
   INTEGER RECVBUF
   INTEGER RECVCOUNTS(*)
   INTEGER RDISPLS(*)
   INTEGER RECVTYPES(*)
   INTEGER COMM
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IALLTOALLW(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPES, &
                           RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPES, &
                           COMM, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Ialltoallw_f2vftr(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPES, &
                                   RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPES, &
                                   COMM, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IALLTOALLW

#endif

#ifdef _MPI

SUBROUTINE MPI_IALLGATHER(SENDBUF, SENDCOUNT, SENDTYPE, &
                          RECVBUF, RECVCOUNT, RECVTYPE, &
                          COMM, REQUEST, ERROR)
   USE vftr_mpi_iallgather_f2vftr_fi, &
      ONLY : vftr_MPI_Iallgather_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IALLGATHER
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNT
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNT
   INTEGER RECVTYPE
   INTEGER COMM
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IALLGATHER(SENDBUF, SENDCOUNT, SENDTYPE, &
                           RECVBUF, RECVCOUNT, RECVTYPE, &
                           COMM, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Iallgather_f2vftr(SENDBUF, SENDCOUNT, SENDTYPE, &
                                   RECVBUF, RECVCOUNT, RECVTYPE, &
                                   COMM, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IALLGATHER

#endif

#ifdef _MPI

SUBROUTINE MPI_IALLTOALL(SENDBUF, SENDCOUNT, SENDTYPE, &
                         RECVBUF, RECVCOUNT, RECVTYPE, &
                         COMM, REQUEST,ERROR)
   USE vftr_mpi_ialltoall_f2vftr_fi, &
      ONLY : vftr_MPI_Ialltoall_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IALLTOALL
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
      CALL PMPI_IALLTOALL(SENDBUF, SENDCOUNT, SENDTYPE, &
                          RECVBUF, RECVCOUNT, RECVTYPE, &
                          COMM, REQUEST,ERROR)
   ELSE
      CALL vftr_MPI_Ialltoall_f2vftr(SENDBUF, SENDCOUNT, SENDTYPE, &
                                  RECVBUF, RECVCOUNT, RECVTYPE, &
                                  COMM, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IALLTOALL

#endif

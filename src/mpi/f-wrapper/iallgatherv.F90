#ifdef _MPI

SUBROUTINE MPI_IALLGATHERV(SENDBUF, SENDCOUNT, SENDTYPE, &
                           RECVBUF, RECVCOUNTS, DISPLS, &
                           RECVTYPE, COMM, REQUEST, ERROR)
   USE vftr_mpi_iallgatherv_f2vftr_fi, &
      ONLY : vftr_MPI_Iallgatherv_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IALLGATHERV
   IMPLICIT NONE
   INTEGER ::  SENDBUF
   INTEGER ::  SENDCOUNT
   INTEGER ::  SENDTYPE
   INTEGER ::  RECVBUF
   INTEGER ::  RECVCOUNTS(*)
   INTEGER ::  DISPLS(*)
   INTEGER ::  RECVTYPE
   INTEGER ::  COMM
   INTEGER ::  REQUEST
   INTEGER ::  ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IALLGATHERV(SENDBUF, SENDCOUNT, SENDTYPE, &
                            RECVBUF, RECVCOUNTS, DISPLS, &
                            RECVTYPE, COMM, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Iallgatherv_f2vftr(SENDBUF, SENDCOUNT, SENDTYPE, &
                                    RECVBUF, RECVCOUNTS, DISPLS, &
                                    RECVTYPE, COMM, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IALLGATHERV

#endif

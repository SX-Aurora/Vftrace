#ifdef _MPI

SUBROUTINE MPI_IGATHERV(SENDBUF, SENDCOUNT, SENDTYPE, &
                        RECVBUF, RECVCOUNTS, DISPLS, &
                        RECVTYPE, ROOT, COMM, &
                        REQUEST, ERROR)
   USE vftr_mpi_igatherv_f2vftr_fi, &
      ONLY : vftr_MPI_Igatherv_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IGATHERV
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNT
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNTS(*)
   INTEGER DISPLS(*)
   INTEGER RECVTYPE
   INTEGER ROOT
   INTEGER COMM
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IGATHERV(SENDBUF, SENDCOUNT, SENDTYPE, &
                         RECVBUF, RECVCOUNTS, DISPLS, &
                         RECVTYPE, ROOT, COMM, &
                         REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Igatherv_f2vftr(SENDBUF, SENDCOUNT, SENDTYPE, &
                                 RECVBUF, RECVCOUNTS, DISPLS, &
                                 RECVTYPE, ROOT, COMM, &
                                 REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IGATHERV

#endif

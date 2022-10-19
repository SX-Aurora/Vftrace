#ifdef _MPI

SUBROUTINE MPI_ISCATTERV(SENDBUF, SENDCOUNTS, DISPLS, &
                         SENDTYPE, RECVBUF, RECVCOUNT, &
                         RECVTYPE, ROOT, COMM, &
                         REQUEST, ERROR)
   USE vftr_mpi_iscatterv_f2vftr_fi, &
      ONLY : vftr_MPI_Iscatterv_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_ISCATTERV
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNTS(*)
   INTEGER DISPLS(*)
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNT
   INTEGER RECVTYPE
   INTEGER ROOT
   INTEGER COMM
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_ISCATTERV(SENDBUF, SENDCOUNTS, DISPLS, &
                          SENDTYPE, RECVBUF, RECVCOUNT, &
                          RECVTYPE, ROOT, COMM, &
                          REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Iscatterv_f2vftr(SENDBUF, SENDCOUNTS, DISPLS, &
                                  SENDTYPE, RECVBUF, RECVCOUNT, &
                                  RECVTYPE, ROOT, COMM, &
                                  REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_ISCATTERV

#endif

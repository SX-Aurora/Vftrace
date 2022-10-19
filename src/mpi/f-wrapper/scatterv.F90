#ifdef _MPI

SUBROUTINE MPI_SCATTERV(SENDBUF, SENDCOUNTS, DISPLS, &
                        SENDTYPE, RECVBUF, RECVCOUNT, &
                        RECVTYPE, ROOT, COMM, &
                        ERROR)
   USE vftr_mpi_scatterv_f2vftr_fi, &
      ONLY : vftr_MPI_Scatterv_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE vftr_sync_time_F, &
      ONLY : vftr_estimate_sync_time
   USE mpi, &
      ONLY : PMPI_SCATTERV
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
   INTEGER ERROR

   CALL vftr_estimate_sync_time("MPI_Scatterv_sync", COMM)

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_SCATTERV(SENDBUF, SENDCOUNTS, DISPLS, &
                         SENDTYPE, RECVBUF, RECVCOUNT, &
                         RECVTYPE, ROOT, COMM, &
                         ERROR)
   ELSE
      CALL vftr_MPI_Scatterv_f2vftr(SENDBUF, SENDCOUNTS, DISPLS, &
                                 SENDTYPE, RECVBUF, RECVCOUNT, &
                                 RECVTYPE, ROOT, COMM, &
                                 ERROR)
   END IF

END SUBROUTINE MPI_SCATTERV

#endif

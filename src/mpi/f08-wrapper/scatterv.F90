#ifdef _MPI

SUBROUTINE MPI_Scatterv_f08(sendbuf, sendcounts, displs, &
                            sendtype, recvbuf, recvcount, &
                            recvtype, root, comm, &
                            error)
   USE vftr_mpi_scatterv_f082vftr_f08i, &
      ONLY : vftr_MPI_Scatterv_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE vftr_sync_time_F08, &
      ONLY : vftr_estimate_sync_time
   USE mpi_f08, &
      ONLY : PMPI_Scatterv, &
             MPI_Datatype, &
             MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcounts(*)
   INTEGER, INTENT(IN) :: displs(*)
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcount
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   INTEGER, INTENT(IN) :: root
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_estimate_sync_time("MPI_Scatterv_sync", comm)

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Scatterv(sendbuf, sendcounts, displs, &
                         sendtype, recvbuf, recvcount, &
                         recvtype, root, comm, &
                         tmperror)
   ELSE
      CALL vftr_MPI_Scatterv_f082vftr(sendbuf, sendcounts, displs, &
                                   sendtype%MPI_VAL, recvbuf, recvcount, &
                                   recvtype%MPI_VAL, root, comm%MPI_VAL, &
                                   tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Scatterv_f08

#endif

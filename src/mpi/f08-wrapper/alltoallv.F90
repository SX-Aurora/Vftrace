#ifdef _MPI

SUBROUTINE MPI_Alltoallv_f08(sendbuf, sendcounts, sdispls, sendtype, &
                             recvbuf, recvcounts, rdispls, recvtype, &
                             comm, error)
   USE vftr_mpi_alltoallv_f082vftr_f08i, &
      ONLY : vftr_MPI_Alltoallv_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE vftr_sync_time_F08, &
      ONLY : vftr_estimate_sync_time
   USE mpi_f08, &
      ONLY : PMPI_Alltoallv, &
             MPI_Datatype, &
             MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcounts(*)
   INTEGER, INTENT(IN) :: sdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcounts(*)
   INTEGER, INTENT(IN) :: rdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_estimate_sync_time("MPI_Alltoallv_sync", comm)

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, &
                          recvbuf, recvcounts, rdispls, recvtype, &
                          comm, tmperror)
   ELSE
      CALL vftr_MPI_Alltoallv_f082vftr(sendbuf, sendcounts, sdispls, sendtype%MPI_VAL, &
                                    recvbuf, recvcounts, rdispls, recvtype%MPI_VAL, &
                                    comm%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Alltoallv_f08

#endif

#ifdef _MPI

SUBROUTINE MPI_Ialltoallv_f08(sendbuf, sendcounts, sdispls, sendtype, &
                              recvbuf, recvcounts, rdispls, recvtype, &
                              comm, request, error)
   USE vftr_mpi_ialltoallv_f082vftr_f08i, &
      ONLY : vftr_MPI_Ialltoallv_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Ialltoallv, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Request
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
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Ialltoallv(sendbuf, sendcounts, sdispls, sendtype, &
                           recvbuf, recvcounts, rdispls, recvtype, &
                           comm, request, tmperror)
   ELSE
      CALL vftr_MPI_Ialltoallv_f082vftr(sendbuf, sendcounts, sdispls, sendtype%MPI_VAL, &
                                        recvbuf, recvcounts, rdispls, recvtype%MPI_VAL, &
                                        comm%MPI_VAL, request%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Ialltoallv_f08

#endif

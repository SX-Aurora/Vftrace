#ifdef _MPI

SUBROUTINE MPI_Ialltoallw_f08(sendbuf, sendcounts, sdispls, sendtypes, &
                              recvbuf, recvcounts, rdispls, recvtypes, &
                              comm, request, error)
   USE vftr_mpi_ialltoallw_f082vftr_f08i, &
      ONLY : vftr_MPI_Ialltoallw_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Ialltoallw, &
             PMPI_Comm_test_inter, &
             PMPI_Comm_remote_size, &
             PMPI_Comm_size, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Request
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcounts(*)
   INTEGER, INTENT(IN) :: sdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: sendtypes(*)
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcounts(*)
   INTEGER, INTENT(IN) :: rdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: recvtypes(*)
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror
   INTEGER, DIMENSION(:), ALLOCATABLE :: tmpsendtypes
   INTEGER, DIMENSION(:), ALLOCATABLE :: tmprecvtypes
   INTEGER :: comm_size, i
   LOGICAL :: isintercom

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes, &
                           recvbuf, recvcounts, rdispls, recvtypes, &
                           comm, request, tmperror)
   ELSE
      CALL PMPI_Comm_test_inter(comm, isintercom, tmperror)
      IF (isintercom) THEN
         CALL PMPI_Comm_remote_size(comm, comm_size, tmperror)
      ELSE
         CALL PMPI_Comm_size(comm, comm_size, tmperror)
      END IF

      ALLOCATE(tmpsendtypes(comm_size))
      ALLOCATE(tmprecvtypes(comm_size))
      DO i = 1, comm_size
         tmpsendtypes(i) = sendtypes(i)%MPI_VAL
      END DO
      DO i = 1, comm_size
         tmprecvtypes(i) = recvtypes(i)%MPI_VAL
      END DO

      CALL vftr_MPI_Ialltoallw_f082vftr(sendbuf, sendcounts, sdispls, tmpsendtypes, &
                                        recvbuf, recvcounts, rdispls, tmprecvtypes, &
                                        comm%MPI_VAL, request%MPI_VAL, tmperror)

      DEALLOCATE(tmpsendtypes)
      DEALLOCATE(tmprecvtypes)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Ialltoallw_f08

#endif

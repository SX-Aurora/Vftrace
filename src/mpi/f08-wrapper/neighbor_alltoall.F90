#ifdef _MPI

SUBROUTINE MPI_Neighbor_alltoall_f08(sendbuf, sendcount, sendtype, &
                                     recvbuf, recvcount, recvtype, &
                                     comm, error)
   USE vftr_mpi_neighbor_alltoall_f082vftr_f08i, &
      ONLY : vftr_MPI_Neighbor_alltoall_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE vftr_sync_time_F08, &
      ONLY : vftr_estimate_sync_time
   USE mpi_f08, &
      ONLY : PMPI_Neighbor_alltoall, &
             MPI_Datatype, &
             MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcount
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcount
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_estimate_sync_time("MPI_Neighbor_alltoall_sync", comm)

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Neighbor_alltoall(sendbuf, sendcount, sendtype, &
                                  recvbuf, recvcount, recvtype, &
                                  comm, tmperror)
   ELSE
      CALL vftr_MPI_Neighbor_alltoall_f082vftr(sendbuf, sendcount, sendtype%MPI_VAL, &
                                               recvbuf, recvcount, recvtype%MPI_VAL, &
                                               comm%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Neighbor_alltoall_f08

#endif

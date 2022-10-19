#ifdef _MPI

SUBROUTINE MPI_Ireduce_scatter_f08(sendbuf, recvbuf, recvcounts, &
                                   datatype, op, comm, &
                                   request, error)
   USE vftr_mpi_ireduce_scatter_f082vftr_f08i, &
      ONLY : vftr_MPI_Ireduce_scatter_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Ireduce_scatter, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Op, &
             MPI_Request
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcounts(*)
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   TYPE(MPI_Op), INTENT(IN) :: op
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Ireduce_scatter(sendbuf, recvbuf, recvcounts, &
                                datatype, op, comm, &
                                request, tmperror)
   ELSE
      CALL vftr_MPI_Ireduce_scatter_f082vftr(sendbuf, recvbuf, recvcounts, &
                                             datatype%MPI_VAL, op%MPI_VAL, comm%MPI_VAL, &
                                             request%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Ireduce_scatter_f08

#endif

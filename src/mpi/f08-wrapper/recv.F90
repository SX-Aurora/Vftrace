#ifdef _MPI

SUBROUTINE MPI_Recv_f08(buf, count, datatype, source, tag, comm, status, error)
   USE vftr_mpi_recv_f082vftr_f08i, &
      ONLY : vftr_MPI_Recv_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Recv, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Status
   IMPLICIT NONE
   INTEGER BUF
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: source
   INTEGER, INTENT(IN) :: tag
   TYPE(MPI_Comm) comm
   TYPE(MPI_Status) :: status
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Recv(buf, count, datatype, source, tag, comm, status, tmperror)
   ELSE
      CALL vftr_MPI_Recv_f082vftr(buf, count, datatype%MPI_VAL, source, tag, comm%MPI_VAL, status, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Recv_f08

#endif

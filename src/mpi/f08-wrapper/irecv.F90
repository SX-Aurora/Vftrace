#ifdef _MPI

SUBROUTINE MPI_Irecv_f08(buf, count, datatype, source, tag, comm, request, error)
   USE vftr_mpi_irecv_f082vftr_f08i, &
      ONLY : vftr_MPI_Irecv_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Irecv, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Request
   IMPLICIT NONE
   INTEGER BUF
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: source
   INTEGER, INTENT(IN) :: tag
   TYPE(MPI_Comm) comm
   TYPE(MPI_Request) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Irecv(buf, count, datatype, source, tag, comm, request, tmperror)
   ELSE
      CALL vftr_MPI_Irecv_f082vftr(buf, count, datatype%MPI_VAL, source, tag, comm%MPI_VAL, request%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Irecv_f08

#endif

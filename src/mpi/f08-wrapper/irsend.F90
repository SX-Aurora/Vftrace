#ifdef _MPI

SUBROUTINE MPI_Irsend_f08(buf, count, datatype, dest, tag, comm, request, error)
   USE vftr_mpi_irsend_f082vftr_f08i, &
      ONLY : vftr_MPI_Irsend_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Irsend, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Request
   IMPLICIT NONE
   INTEGER :: buf
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: dest
   INTEGER, INTENT(IN) :: tag
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Irsend(buf, count, datatype, dest, tag, comm, request, tmperror)
   ELSE
      CALL vftr_MPI_Irsend_f082vftr(buf, count, datatype%MPI_VAL, dest, tag, comm%MPI_VAL, request%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Irsend_f08

#endif

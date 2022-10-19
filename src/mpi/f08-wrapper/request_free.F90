#ifdef _MPI

SUBROUTINE MPI_Request_free_f08(request, error)
   USE vftr_mpi_request_free_f082vftr_f08i, &
      ONLY : vftr_MPI_Request_free_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Request_free, &
             MPI_Request
   IMPLICIT NONE
   TYPE(MPI_Request), INTENT(INOUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Request_free(request, tmperror)
   ELSE
      CALL vftr_MPI_Request_free_f082vftr(request%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Request_free_f08

#endif

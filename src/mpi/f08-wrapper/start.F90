#ifdef _MPI

SUBROUTINE MPI_Start_f08(request, error)
   USE vftr_mpi_start_f082vftr_f08i, &
      ONLY : vftr_MPI_Start_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Start, &
             MPI_Request
   IMPLICIT NONE
   TYPE(MPI_Request), INTENT(INOUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Start(request, tmperror)
   ELSE
      CALL vftr_MPI_Start_f082vftr(request%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Start_f08

#endif

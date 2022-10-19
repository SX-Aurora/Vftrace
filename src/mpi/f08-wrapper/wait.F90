#ifdef _MPI

SUBROUTINE MPI_Wait_f08(request, status, error)
   USE vftr_mpi_wait_f082vftr_f08i, &
      ONLY : vftr_MPI_Wait_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY: PMPI_Wait, &
            MPI_Request, &
            MPI_Status
   IMPLICIT NONE
   TYPE(MPI_Request), INTENT(INOUT) :: request
   TYPE(MPI_Status) :: status
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Wait(request, status, tmperror)
   ELSE
      CALL vftr_MPI_Wait_f082vftr(request%MPI_VAL, status, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Wait_f08

#endif

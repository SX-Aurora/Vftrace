#ifdef _MPI

SUBROUTINE MPI_Ibarrier_f08(comm, request, error)
   USE vftr_mpi_ibarrier_f082vftr_f08i, &
      ONLY : vftr_MPI_Ibarrier_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Barrier, &
             MPI_Comm, &
             MPI_Request
   IMPLICIT NONE
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Ibarrier(comm, request, tmperror)
   ELSE
      CALL vftr_MPI_Ibarrier_f082vftr(comm%MPI_VAL, request%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Ibarrier_f08

#endif

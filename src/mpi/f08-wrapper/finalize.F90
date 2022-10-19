#ifdef _MPI

SUBROUTINE MPI_Finalize_f08(error)
   USE vftr_mpi_finalize_f082vftr_f08i, &
      ONLY : vftr_mpi_finalize_f082vftr
   IMPLICIT NONE
   INTEGER, OPTIONAL, INTENT(OUT) :: error

   INTEGER :: tmperror

   CALL vftr_mpi_finalize_f082vftr(tmperror)

   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Finalize_f08

#endif

#ifdef _MPI

SUBROUTINE MPI_Init_thread_f08(required, provided, error)
   USE vftr_mpi_init_thread_f082vftr_f08i, &
      ONLY : vftr_mpi_init_thread_f082vftr
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: required
   INTEGER, INTENT(OUT) :: provided
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_mpi_init_thread_f082vftr(required, provided, tmperror)

   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Init_thread_f08

#endif

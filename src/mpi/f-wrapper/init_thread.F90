#ifdef _MPI

SUBROUTINE MPI_INIT_THREAD(REQUIRED, PROVIDED, IERROR)
   USE vftr_mpi_init_thread_f2vftr_fi, &
      ONLY : vftr_mpi_init_thread_f2vftr
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: REQUIRED
   INTEGER, INTENT(OUT) :: PROVIDED
   INTEGER, INTENT(OUT) :: IERROR

   CALL vftr_mpi_init_thread_f2vftr(REQUIRED, PROVIDED, IERROR)

END SUBROUTINE MPI_INIT_THREAD

#endif

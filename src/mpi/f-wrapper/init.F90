#ifdef _MPI

SUBROUTINE MPI_INIT(IERROR)
   USE vftr_mpi_init_f2vftr_fi, &
      ONLY : vftr_mpi_init_f2vftr
   IMPLICIT NONE
   INTEGER, INTENT(OUT) :: IERROR

   CALL vftr_mpi_init_f2vftr(IERROR)

END SUBROUTINE MPI_INIT

#endif

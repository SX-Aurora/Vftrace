#ifdef _MPI

SUBROUTINE MPI_FINALIZE(IERROR)
   USE vftr_mpi_finalize_f2vftr_fi, &
      ONLY : vftr_mpi_finalize_f2vftr

   IMPLICIT NONE

   INTEGER, INTENT(OUT) :: IERROR

   CALL vftr_mpi_finalize_f2vftr(IERROR)

END SUBROUTINE MPI_FINALIZE

#endif

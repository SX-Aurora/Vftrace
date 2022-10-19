#ifdef _MPI

SUBROUTINE MPI_PCONTROL(LEVEL)
   USE vftr_mpi_pcontrol_f2vftr_fi, &
      ONLY : vftr_MPI_Pcontrol_f2vftr
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: LEVEL

   CALL vftr_MPI_Pcontrol_f2vftr(LEVEL)

END SUBROUTINE MPI_PCONTROL

#endif

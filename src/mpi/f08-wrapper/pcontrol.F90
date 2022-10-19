#ifdef _MPI

SUBROUTINE MPI_Pcontrol_f08(level)
   USE vftr_mpi_pcontrol_f082vftr_f08i, &
      ONLY : vftr_MPI_Pcontrol_f082vftr
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: level

   CALL vftr_MPI_Pcontrol_f082vftr(level)

END SUBROUTINE MPI_Pcontrol_f08

#endif

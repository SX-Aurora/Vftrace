#ifdef _MPI

SUBROUTINE MPI_START(REQUEST, ERROR)
   USE vftr_mpi_start_f2vftr_fi, &
      ONLY : vftr_MPI_Start_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_START
   IMPLICIT NONE
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_START(REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Start_f2vftr(REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_START

#endif

#ifdef _MPI

SUBROUTINE MPI_Startall(COUNT, ARRAY_OREQUESTS, ERROR)
   USE vftr_mpi_startall_f2vftr_fi, &
      ONLY : vftr_MPI_Startall_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_STARTALL
   IMPLICIT NONE
   INTEGER COUNT
   INTEGER ARRAY_OREQUESTS(*)
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_Startall(COUNT, ARRAY_OREQUESTS, ERROR)
   ELSE
      CALL vftr_MPI_Startall_f2vftr(COUNT, ARRAY_OREQUESTS, ERROR)
   END IF

END SUBROUTINE MPI_Startall

#endif


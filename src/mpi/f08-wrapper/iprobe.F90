#ifdef _MPI

SUBROUTINE MPI_Iprobe_f08(source, tag, comm, flag, status, error)
   USE vftr_mpi_iprobe_f082vftr_f08i, &
      ONLY : vftr_MPI_Iprobe_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Iprobe, &
             MPI_Comm, &
             MPI_Status
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: source
   INTEGER, INTENT(IN) :: tag
   TYPE(MPI_Comm), INTENT(IN) :: comm
   LOGICAL, INTENT(OUT) :: flag
   TYPE(MPI_Status) :: status
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmpflag, tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Iprobe(source, tag, comm, flag, status, tmperror)
   ELSE
      CALL vftr_MPI_Iprobe_f082vftr(source, tag, comm%MPI_VAL, tmpflag, status, tmperror)
      FLAG = (TMPFLAG /= 0)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Iprobe_f08

#endif

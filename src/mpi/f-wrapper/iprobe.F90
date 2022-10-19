#ifdef _MPI

SUBROUTINE MPI_IPROBE(SOURCE, TAG, COMM, FLAG, STATUS, ERROR)
   USE vftr_mpi_iprobe_f2vftr_fi, &
      ONLY : vftr_MPI_Iprobe_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IPROBE, &
             MPI_STATUS_SIZE
   IMPLICIT NONE
   INTEGER SOURCE
   INTEGER TAG
   INTEGER COMM
   LOGICAL FLAG
   INTEGER STATUS(MPI_STATUS_SIZE)
   INTEGER ERROR

   INTEGER TMPFLAG

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IPROBE(SOURCE, TAG, COMM, FLAG, STATUS, ERROR)
   ELSE
      CALL vftr_MPI_Iprobe_f2vftr(SOURCE, TAG, COMM, TMPFLAG, STATUS, ERROR)
      FLAG = (TMPFLAG /= 0)
   END IF

END SUBROUTINE MPI_IPROBE

#endif

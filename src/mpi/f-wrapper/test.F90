#ifdef _MPI

SUBROUTINE MPI_TEST(REQUEST, FLAG, STATUS, ERROR)
   USE vftr_mpi_test_f2vftr_fi, &
      ONLY : vftr_MPI_Test_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY: PMPI_TEST, &
            MPI_STATUS_SIZE
   IMPLICIT NONE
   INTEGER REQUEST
   LOGICAL FLAG
   INTEGER STATUS(MPI_STATUS_SIZE)
   INTEGER ERROR
   INTEGER TMPFLAG

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_TEST(REQUEST, FLAG, STATUS, ERROR)
   ELSE
      CALL vftr_MPI_Test_f2vftr(REQUEST, TMPFLAG, STATUS, ERROR)

      FLAG = (TMPFLAG /= 0)
   END IF

END SUBROUTINE MPI_TEST

#endif

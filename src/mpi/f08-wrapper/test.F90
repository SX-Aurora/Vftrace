#ifdef _MPI

SUBROUTINE MPI_Test_f08(request, flag, status, error)
   USE vftr_mpi_test_f082vftr_f08i, &
      ONLY : vftr_MPI_Test_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY: PMPI_Test, &
            MPI_Request, &
            MPI_Status
   IMPLICIT NONE
   TYPE(MPI_Request), INTENT(INOUT):: request
   LOGICAL, INTENT(OUT) :: flag
   TYPE(MPI_Status) :: status
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER tmpflag, tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Test(request, flag, status, tmperror)
   ELSE
      CALL vftr_MPI_Test_f082vftr(request%MPI_VAL, tmpflag, status, tmperror)

      flag = (tmpflag /= 0)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Test_f08

#endif

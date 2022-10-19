#ifdef _MPI

SUBROUTINE MPI_Testany_f08(count, array_of_requests, index, flag, status, error)
   USE vftr_mpi_testany_f082vftr_f08i, &
      ONLY : vftr_MPI_Testany_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY: PMPI_Testany, &
            MPI_Request, &
            MPI_Status
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Request), INTENT(INOUT) :: array_of_requests(count)
   INTEGER, INTENT(OUT) :: index
   LOGICAL, INTENT(OUT) :: flag
   TYPE(MPI_Status) :: status
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmpflag, tmperror
   INTEGER, DIMENSION(count) :: tmparray_of_requests
   INTEGER :: i

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Testany(count, array_of_requests, index, flag, status, tmperror)
   ELSE
      DO i = 1, count
         tmparray_of_requests(i) = array_of_requests(i)%MPI_VAL
      END DO
      CALL vftr_MPI_Testany_f082vftr(count, tmparray_of_requests, index, tmpflag, status, tmperror)
      DO i = 1, count
         array_of_requests(i)%MPI_VAL = tmparray_of_requests(i)
      END DO
      flag = (tmpflag /= 0)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Testany_f08

#endif

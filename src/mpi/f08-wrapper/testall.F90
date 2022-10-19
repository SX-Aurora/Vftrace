#ifdef _MPI

SUBROUTINE MPI_Testall_f08(count, array_of_requests, flag, array_of_statuses, error)
   USE vftr_mpi_testall_f082vftr_f08i, &
      ONLY : vftr_MPI_Testall_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY: PMPI_Testall, &
            MPI_Request, &
            MPI_Status
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Request), INTENT(INOUT) :: array_of_requests(count)
   LOGICAL :: flag
   TYPE(MPI_Status) :: array_of_statuses(*)
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmpflag, tmperror
   INTEGER, DIMENSION(count) :: tmparray_of_requests
   INTEGER :: i

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Testall(count, array_of_requests, flag, array_of_statuses, tmperror)
   ELSE
      DO i = 1, count
         tmparray_of_requests(i) = array_of_requests(i)%MPI_VAL
      END DO
      CALL vftr_MPI_Testall_f082vftr(count, tmparray_of_requests, tmpflag, array_of_statuses, tmperror)
      DO i = 1, count
         array_of_requests(i)%MPI_VAL = tmparray_of_requests(i)
      END DO
      FLAG = (TMPFLAG /= 0)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Testall_f08

#endif

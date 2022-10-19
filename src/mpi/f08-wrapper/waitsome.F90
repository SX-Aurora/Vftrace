#ifdef _MPI

SUBROUTINE MPI_Waitsome_f08(incount, array_of_requests, outcount, &
                            array_of_indices, array_of_statuses, error)
   USE vftr_mpi_waitsome_f082vftr_f08i, &
      ONLY : vftr_MPI_Waitsome_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY: PMPI_Waitsome, &
            MPI_Request, &
            MPI_Status
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: incount
   TYPE(MPI_Request), INTENT(INOUT) :: array_of_requests(incount)
   INTEGER, INTENT(OUT) :: outcount
   INTEGER, INTENT(OUT) :: array_of_indices(*)
   TYPE(MPI_Status) :: array_of_statuses(*)
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror
   INTEGER, DIMENSION(incount) :: tmparray_of_requests
   INTEGER :: i

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Waitsome(incount, array_of_requests, outcount, &
                         array_of_indices, array_of_statuses, tmperror)
   ELSE
      DO i = 1, incount
         tmparray_of_requests(i) = array_of_requests(i)%MPI_VAL
      END DO
      CALL vftr_MPI_Waitsome_f082vftr(incount, tmparray_of_requests, outcount, &
                                      array_of_indices, array_of_statuses, tmperror)
      DO i = 1, incount
         array_of_requests(i)%MPI_VAL = tmparray_of_requests(i)
      END DO
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Waitsome_f08

#endif

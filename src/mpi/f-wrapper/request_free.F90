#ifdef _MPI

SUBROUTINE MPI_REQUEST_FREE(REQUEST, ERROR)
   USE vftr_mpi_request_free_f2vftr_fi, &
      ONLY : vftr_MPI_Request_free_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_REQUEST_FREE
   IMPLICIT NONE
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_REQUEST_FREE(REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Request_free_f2vftr(REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_REQUEST_FREE

#endif

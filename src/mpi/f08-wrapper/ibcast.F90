#ifdef _MPI

SUBROUTINE MPI_Ibcast_f08(buffer, count, datatype, &
                          root, comm, request, error)
   USE vftr_mpi_ibcast_f082vftr_f08i, &
      ONLY : vftr_MPI_Ibcast_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Ibcast, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Request
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: buffer
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: root
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Ibcast(buffer, count, datatype, &
                       root, comm, request, tmperror)
   ELSE
      CALL vftr_MPI_Ibcast_f082vftr(buffer, count, datatype%MPI_VAL, &
                                    root, comm%MPI_VAL, request%MPI_VAL, &
                                    tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Ibcast_f08

#endif

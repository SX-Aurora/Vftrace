#ifdef _MPI

SUBROUTINE MPI_Ssend_f08(buf, count, datatype, dest, tag, comm, error)
   USE vftr_mpi_ssend_f082vftr_f08i, &
      ONLY : vftr_MPI_Ssend_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Ssend, &
             MPI_Datatype, &
             MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: buf
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: dest
   INTEGER, INTENT(IN) :: tag
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Ssend(buf, count, datatype, dest, tag, comm, tmperror)
   ELSE
      CALL vftr_MPI_Ssend_f082vftr(buf, count, datatype%MPI_VAL, dest, tag, comm%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Ssend_f08

#endif

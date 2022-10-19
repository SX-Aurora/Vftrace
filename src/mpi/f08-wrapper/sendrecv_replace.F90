#ifdef _MPI

SUBROUTINE MPI_Sendrecv_replace_f08(buf, count, datatype, dest, sendtag, source, &
                                    recvtag, comm, status, error)
   USE vftr_mpi_sendrecv_replace_f082vftr_f08i, &
      ONLY : vftr_MPI_Sendrecv_replace_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Sendrecv_replace, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Status
   IMPLICIT NONE
   INTEGER buf
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: dest
   INTEGER, INTENT(IN) :: sendtag
   INTEGER, INTENT(IN) :: source
   INTEGER, INTENT(IN) :: recvtag
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Status) :: status
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, &
                                 recvtag, comm, status, tmperror)
   ELSE
      CALL vftr_MPI_Sendrecv_replace_f082vftr(buf, count, datatype%MPI_VAL, dest, sendtag, source, &
                                              recvtag, comm%MPI_VAL, status, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror


END SUBROUTINE MPI_Sendrecv_replace_f08

#endif

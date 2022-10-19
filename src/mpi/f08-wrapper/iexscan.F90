#ifdef _MPI

SUBROUTINE MPI_Iexscan_f08(sendbuf, recvbuf, count, datatype, &
                           op, comm, request, error)
   USE vftr_mpi_iexscan_f082vftr_f08i, &
      ONLY : vftr_MPI_Iexscan_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Scan, &
             MPI_Datatype, &
             MPI_Op, &
             MPI_Comm, &
             MPI_Request
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   TYPE(MPI_Op), INTENT(IN) :: op
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Iexscan(sendbuf, recvbuf, count, datatype, &
                        op, comm, request, tmperror)
   ELSE
      CALL vftr_MPI_Iexscan_f082vftr(sendbuf, recvbuf, count, datatype%MPI_VAL, &
                                     op%MPI_VAL, comm%MPI_VAL, request%MPI_VAL, &
                                     tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Iexscan_f08

#endif

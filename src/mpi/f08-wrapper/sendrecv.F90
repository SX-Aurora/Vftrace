#ifdef _MPI

SUBROUTINE MPI_Sendrecv_f08(sendbuf, sendcount, sendtype, dest, sendtag, &
                            recvbuf, recvcount, recvtype, source, recvtag, &
                            comm, status, error)
   USE vftr_mpi_sendrecv_f082vftr_f08i, &
      ONLY : vftr_MPI_Sendrecv_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Sendrecv, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Status
   IMPLICIT NONE
   INTEGER :: sendbuf
   INTEGER, INTENT(IN) :: sendcount
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER, INTENT(IN) :: dest
   INTEGER, INTENT(IN) :: sendtag
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcount
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   INTEGER, INTENT(IN) ::  source
   INTEGER, INTENT(IN) ::  recvtag
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Status) :: status
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, &
                         recvbuf, recvcount, recvtype, source, recvtag, &
                         comm, status, tmperror)
   ELSE
      CALL vftr_MPI_Sendrecv_f082vftr(sendbuf, sendcount, sendtype%MPI_VAL, dest, sendtag, &
                                      recvbuf, recvcount, recvtype%MPI_VAL, source, recvtag, &
                                      comm%MPI_VAL, status, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Sendrecv_f08

#endif

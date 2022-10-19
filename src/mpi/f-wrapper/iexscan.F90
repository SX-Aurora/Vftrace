#ifdef _MPI

SUBROUTINE MPI_IEXSCAN(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                       OP, COMM, REQUEST, ERROR)
   USE vftr_mpi_iexscan_f2vftr_fi, &
      ONLY : vftr_MPI_Iexscan_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IEXSCAN
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER OP
   INTEGER COMM
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IEXSCAN(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                        OP, COMM, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Iexscan_f2vftr(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                                OP, COMM, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IEXSCAN

#endif

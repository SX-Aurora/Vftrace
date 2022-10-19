#ifdef _MPI

SUBROUTINE MPI_IALLREDUCE(SENDBUF, RECVBUF, COUNT, &
                          DATATYPE, OP, COMM, &
                          REQUEST, ERROR)
   USE vftr_mpi_iallreduce_f2vftr_fi, &
      ONLY : vftr_MPI_Iallreduce_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IALLREDUCE
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
      CALL PMPI_IALLREDUCE(SENDBUF, RECVBUF, COUNT, &
                           DATATYPE, OP, COMM, &
                           REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Iallreduce_f2vftr(SENDBUF, RECVBUF, COUNT, &
                                   DATATYPE, OP, COMM, &
                                   REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IALLREDUCE

#endif

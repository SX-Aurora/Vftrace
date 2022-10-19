#ifdef _MPI

SUBROUTINE MPI_IREDUCE(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                       OP, ROOT, COMM, REQUEST, ERROR)
   USE vftr_mpi_ireduce_f2vftr_fi, &
      ONLY : vftr_MPI_Ireduce_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IREDUCE
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER OP
   INTEGER ROOT
   INTEGER COMM
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IREDUCE(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                        OP, ROOT, COMM, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Ireduce_f2vftr(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                                OP, ROOT, COMM, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IREDUCE

#endif

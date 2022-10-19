#ifdef _MPI

SUBROUTINE MPI_IGATHER(SENDBUF, SENDCOUNT, SENDTYPE, &
                       RECVBUF, RECVCOUNT, RECVTYPE, &
                       ROOT, COMM, REQUEST, ERROR)
   USE vftr_mpi_igather_f2vftr_fi, &
      ONLY : vftr_MPI_Igather_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IGATHER
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNT
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNT
   INTEGER RECVTYPE
   INTEGER ROOT
   INTEGER COMM
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IGATHER(SENDBUF, SENDCOUNT, SENDTYPE, &
                        RECVBUF, RECVCOUNT, RECVTYPE, &
                        ROOT, COMM, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Igather_f2vftr(SENDBUF, SENDCOUNT, SENDTYPE, &
                                RECVBUF, RECVCOUNT, RECVTYPE, &
                                ROOT, COMM, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IGATHER

#endif

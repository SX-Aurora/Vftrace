#ifdef _MPI

SUBROUTINE MPI_IBCAST(BUFFER, COUNT, DATATYPE, &
                      ROOT, COMM, REQUEST, ERROR)
   USE vftr_mpi_ibcast_f2vftr_fi, &
      ONLY : vftr_MPI_Ibcast_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IBCAST
   IMPLICIT NONE
   INTEGER BUFFER
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER ROOT
   INTEGER COMM
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IBCAST(BUFFER, COUNT, DATATYPE, &
                       ROOT, COMM, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Ibcast_f2vftr(BUFFER, COUNT, DATATYPE, &
                               ROOT, COMM, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IBCAST

#endif

#ifdef _MPI

SUBROUTINE MPI_Igather_f08(sendbuf, sendcount, sendtype, &
                          recvbuf, recvcount, recvtype, &
                          root, comm, request, error)
   USE vftr_mpi_igather_f082vftr_f08i, &
      ONLY : vftr_MPI_Igather_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Igather, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Request
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcount
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcount
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   INTEGER, INTENT(IN) :: root
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Igather(sendbuf, sendcount, sendtype, &
                        recvbuf, recvcount, recvtype, &
                        root, comm, request, tmperror)
   ELSE
      CALL vftr_MPI_Igather_f082vftr(sendbuf, sendcount, sendtype%MPI_VAL, &
                                     recvbuf, recvcount, recvtype%MPI_VAL, &
                                     root, comm%MPI_VAL, request%MPI_VAL, &
                                     tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Igather_f08

#endif

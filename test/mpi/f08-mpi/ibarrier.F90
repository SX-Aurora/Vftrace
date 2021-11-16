PROGRAM ibarrier
   USE mpi_f08
   IMPLICIT NONE
   INTEGER :: ierr
   TYPE(MPI_Request) :: request
   TYPE(MPI_Status) :: status

   CALL MPI_Init(ierr)
   CALL MPI_Ibarrier(MPI_COMM_WORLD, request, ierr)
   CALL MPI_Wait(request, status, ierr)
   CALL MPI_Finalize(ierr)
END PROGRAM ibarrier

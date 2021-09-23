PROGRAM ibarrier
   USE mpi
   IMPLICIT NONE
   INTEGER :: ierr
   INTEGER :: request
   INTEGER :: status(MPI_STATUS_SIZE)

   CALL MPI_Init(ierr)
   CALL MPI_Ibarrier(MPI_COMM_WORLD, request, ierr)
   CALL MPI_Wait(request, status, ierr)
   CALL MPI_Finalize(ierr)
END PROGRAM ibarrier

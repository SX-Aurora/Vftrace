PROGRAM ibarrier
   USE mpi
   IMPLICIT NONE
   INTEGER :: ierr
   INTEGER :: request

   CALL MPI_Init(ierr)
   CALL MPI_Ibarrier(MPI_COMM_WORLD, request, ierr)
   CALL MPI_Wait(request, ierr)
   CALL MPI_Finalize(ierr)
END PROGRAM ibarrier

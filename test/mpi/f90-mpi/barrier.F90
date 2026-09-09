PROGRAM Barrier
   USE mpi
   IMPLICIT NONE
   INTEGER :: ierr

   CALL MPI_Init(ierr)
   CALL MPI_Barrier(MPI_COMM_WORLD, ierr)
   CALL MPI_Finalize(ierr)
END PROGRAM Barrier

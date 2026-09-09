PROGRAM init_finalize
   USE mpi
   IMPLICIT NONE
   INTEGER :: ierr

   CALL MPI_Init(ierr)
   CALL MPI_Finalize(ierr)
END PROGRAM init_finalize

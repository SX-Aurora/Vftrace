PROGRAM init_thread_finalize
   USE mpi
   IMPLICIT NONE
   INTEGER :: provided
   INTEGER :: ierr

   CALL MPI_Init_thread(MPI_THREAD_SINGLE, provided, ierr)
   CALL MPI_Finalize(ierr)
END PROGRAM init_thread_finalize

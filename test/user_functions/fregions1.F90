PROGRAM fregions1
#ifdef _MPI
   USE mpi
#endif

   USE vftrace

   IMPLICIT NONE

#ifdef _MPI
   INTEGER :: ierr

   CALL MPI_Init(ierr)
#endif

   CALL vftrace_region_begin("user-region-1")
   CALL vftrace_region_end("user-region-1")

#ifdef _MPI
   CALL MPI_Finalize(ierr)
#endif
END PROGRAM fregions1

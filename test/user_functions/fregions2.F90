PROGRAM fregions2
#ifdef _MPI
   USE mpi
#endif

   USE vftrace

   IMPLICIT NONE

   CHARACTER(LEN=16) :: reg_name

#ifdef _MPI
   INTEGER :: ierr

   CALL MPI_Init(ierr)
#endif

   reg_name = "user-region-1"
   CALL vftrace_region_begin(reg_name)
   CALL vftrace_region_end(reg_name)

#ifdef _MPI
   CALL MPI_Finalize(ierr)
#endif
END PROGRAM fregions2

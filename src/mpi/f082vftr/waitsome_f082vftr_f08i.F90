MODULE vftr_mpi_waitsome_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Waitsome_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Waitsome_f082vftr(f_incount, f_array_of_requests, &
                                            f_outcount, f_array_of_indices, &
                                            f_array_of_statuses, f_error) &
         BIND(C, name="vftr_MPI_Waitsome_f082vftr")
         USE mpi_f08, &
            ONLY: MPI_Status
         IMPLICIT NONE
         INTEGER :: f_incount
         INTEGER :: f_array_of_requests(*)
         INTEGER :: f_outcount
         INTEGER :: f_array_of_indices(*)
         TYPE(MPI_Status) :: f_array_of_statuses(*)
         INTEGER :: f_error
      END SUBROUTINE vftr_MPI_Waitsome_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_waitsome_f082vftr_f08i

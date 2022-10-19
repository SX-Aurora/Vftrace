MODULE vftr_mpi_startall_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Startall_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Startall_f082vftr(f_count, f_array_of_requests, &
                                            f_error) &
         BIND(C, name="vftr_MPI_Startall_f082vftr")
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: f_count
         INTEGER, INTENT(INOUT) :: f_array_of_requests(*)
         INTEGER, INTENT(OUT) :: f_error
      END SUBROUTINE vftr_MPI_Startall_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_startall_f082vftr_f08i

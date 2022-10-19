MODULE vftr_mpi_test_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Test_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Test_f082vftr(f_request, f_flag, f_status, f_error) &
         BIND(C, name="vftr_MPI_Test_f082vftr")
         USE mpi_f08, &
            ONLY: MPI_Status
         IMPLICIT NONE
         INTEGER :: f_request
         INTEGER :: f_flag
         TYPE(MPI_Status) :: f_status
         INTEGER :: f_error
      END SUBROUTINE vftr_MPI_Test_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_test_f082vftr_f08i

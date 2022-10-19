MODULE vftr_mpi_init_thread_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Init_thread_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Init_thread_f082vftr(required, provided, error) &
         BIND(c, NAME="vftr_MPI_Init_thread_f082vftr")
         IMPLICIT NONE
         INTEGER :: required
         INTEGER :: provided
         INTEGER :: error
      END SUBROUTINE vftr_MPI_Init_thread_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_init_thread_f082vftr_f08i

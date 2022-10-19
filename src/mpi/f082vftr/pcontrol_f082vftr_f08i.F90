MODULE vftr_mpi_pcontrol_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Pcontrol_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Pcontrol_f082vftr(level) &
         BIND(c, NAME="vftr_MPI_Pcontrol_f082vftr")
         USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int
         IMPLICIT NONE
         INTEGER(KIND=c_int), VALUE, INTENT(IN) :: level
      END SUBROUTINE vftr_MPI_Pcontrol_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_pcontrol_f082vftr_f08i

MODULE vftr_mpi_pcontrol_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Pcontrol_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Pcontrol_f2vftr(level) &
         BIND(c, NAME="vftr_MPI_Pcontrol_f2vftr")
         USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int
         IMPLICIT NONE
         INTEGER(KIND=c_int), VALUE, INTENT(IN) :: level
      END SUBROUTINE vftr_MPI_Pcontrol_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_pcontrol_f2vftr_fi

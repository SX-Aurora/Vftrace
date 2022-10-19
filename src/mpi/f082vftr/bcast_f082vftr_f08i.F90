MODULE vftr_mpi_bcast_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Bcast_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Bcast_f082vftr(buffer, count, f_datatype, &
                                         root, f_comm, f_error) &
         BIND(C, name="vftr_MPI_Bcast_f082vftr")
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: buffer
         INTEGER, INTENT(IN) :: count
         INTEGER, INTENT(IN) :: f_datatype
         INTEGER, INTENT(IN) :: root
         INTEGER, INTENT(IN) :: f_comm
         INTEGER, INTENT(OUT) :: f_error
      END SUBROUTINE vftr_MPI_Bcast_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_bcast_f082vftr_f08i

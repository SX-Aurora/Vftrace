MODULE vftr_mpi_ibsend_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Ibsend_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Ibsend_f082vftr(buf, count, f_datatype, dest, tag, &
                                          f_comm, f_request, f_error) &
         BIND(C, name="vftr_MPI_Ibsend_f082vftr")
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: buf
         INTEGER, INTENT(IN) :: count
         INTEGER, INTENT(IN) :: f_datatype
         INTEGER, INTENT(IN) :: dest
         INTEGER, INTENT(IN) :: tag
         INTEGER, INTENT(IN) :: f_comm
         INTEGER, INTENT(IN) :: f_request
         INTEGER, INTENT(OUT) :: f_error
      END SUBROUTINE vftr_MPI_Ibsend_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_ibsend_f082vftr_f08i

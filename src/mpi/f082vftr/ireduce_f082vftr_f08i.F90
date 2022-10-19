MODULE vftr_mpi_ireduce_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Ireduce_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Ireduce_f082vftr(sendbuf, recvbuf, count, f_datatype, &
                                           f_op, root, f_comm, f_request, f_error) &
         BIND(C, name="vftr_MPI_Ireduce_f082vftr")
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: sendbuf
         INTEGER, INTENT(IN) :: recvbuf
         INTEGER, INTENT(IN) :: count
         INTEGER, INTENT(IN) :: f_datatype
         INTEGER, INTENT(IN) :: f_op
         INTEGER, INTENT(IN) :: root
         INTEGER, INTENT(IN) :: f_comm
         INTEGER, INTENT(OUT) :: f_request
         INTEGER, INTENT(OUT) :: f_error
      END SUBROUTINE vftr_MPI_Ireduce_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_ireduce_f082vftr_f08i

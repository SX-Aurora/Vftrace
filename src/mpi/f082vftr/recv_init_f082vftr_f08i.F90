MODULE vftr_mpi_recv_init_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Recv_init_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Recv_init_f082vftr(buf, count, f_datatype, source, tag, &
                                             f_comm, f_request, f_error) &
         BIND(C, name="vftr_MPI_Recv_init_f082vftr")
         IMPLICIT NONE
         INTEGER :: buf
         INTEGER :: count
         INTEGER :: f_datatype
         INTEGER :: source
         INTEGER :: tag
         INTEGER :: f_comm
         INTEGER :: f_request
         INTEGER :: f_error
      END SUBROUTINE vftr_MPI_Recv_init_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_recv_init_f082vftr_f08i

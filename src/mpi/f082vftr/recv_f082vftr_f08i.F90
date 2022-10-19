MODULE vftr_mpi_recv_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Recv_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Recv_f082vftr(buf, count, f_datatype, source, tag, &
                                        f_comm, f_status, f_error) &
         BIND(C, name="vftr_MPI_Recv_f082vftr")
         USE mpi_f08, &
            ONLY : MPI_Status
         IMPLICIT NONE
         INTEGER :: buf
         INTEGER :: count
         INTEGER :: f_datatype
         INTEGER :: source
         INTEGER :: tag
         INTEGER :: f_comm
         TYPE(MPI_Status) :: f_status
         INTEGER :: f_error
      END SUBROUTINE vftr_MPI_Recv_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_recv_f082vftr_f08i

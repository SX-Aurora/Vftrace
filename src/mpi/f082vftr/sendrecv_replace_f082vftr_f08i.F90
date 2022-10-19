MODULE vftr_mpi_sendrecv_replace_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Sendrecv_replace_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Sendrecv_replace_f082vftr(buf, count, f_datatype, dest, sendtag, source, &
                                                    recvtag, f_comm, f_status, f_error) &
         BIND(C, name="vftr_MPI_Sendrecv_replace_f082vftr")
         USE mpi_f08, &
            ONLY : MPI_Status
         IMPLICIT NONE
         INTEGER :: buf
         INTEGER :: count
         INTEGER :: f_datatype
         INTEGER :: dest
         INTEGER :: sendtag
         INTEGER :: source
         INTEGER :: recvtag
         INTEGER :: f_comm
         TYPE(MPI_Status) :: f_status
         INTEGER :: f_error
      END SUBROUTINE vftr_MPI_Sendrecv_replace_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_sendrecv_replace_f082vftr_f08i

MODULE vftr_mpi_reduce_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Reduce_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Reduce_f2vftr(SENDBUF, RECVBUF, COUNT, F_DATATYPE, &
                                     F_OP, ROOT, F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Reduce_f2vftr")
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER RECVBUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER F_OP
         INTEGER ROOT
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Reduce_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_reduce_f2vftr_fi

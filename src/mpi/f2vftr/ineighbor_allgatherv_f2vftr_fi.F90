MODULE vftr_mpi_ineighbor_allgatherv_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Ineighbor_allgatherv_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Ineighbor_allgatherv_f2vftr(SENDBUF, SENDCOUNT, F_SENDTYPE, &
                                                      RECVBUF, F_RECVCOUNTS, F_DISPLS, &
                                                      F_RECVTYPE, F_COMM, F_REQUEST, &
                                                      F_ERROR) &
         BIND(C, name="vftr_MPI_Ineighbor_allgatherv_f2vftr")
         IMPLICIT NONE
         INTEGER :: SENDBUF
         INTEGER :: SENDCOUNT
         INTEGER :: F_SENDTYPE
         INTEGER :: RECVBUF
         INTEGER :: F_RECVCOUNTS(*)
         INTEGER :: F_DISPLS(*)
         INTEGER :: F_RECVTYPE
         INTEGER :: F_COMM
         INTEGER :: F_REQUEST
         INTEGER :: F_ERROR
      END SUBROUTINE vftr_MPI_Ineighbor_allgatherv_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_ineighbor_allgatherv_f2vftr_fi

MODULE vftr_mpi_neighbor_alltoallv_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Neighbor_alltoallv_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Neighbor_alltoallv_f2vftr(SENDBUF, F_SENDCOUNTS, &
                                                    F_SDISPLS, F_SENDTYPE, &
                                                    RECVBUF, F_RECVCOUNTS, &
                                                    F_RDISPLS, F_RECVTYPE, &
                                                    F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Neighbor_alltoallv_f2vftr")
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER F_SENDCOUNTS(*)
         INTEGER F_SDISPLS(*)
         INTEGER F_SENDTYPE
         INTEGER RECVBUF
         INTEGER F_RECVCOUNTS(*)
         INTEGER F_RDISPLS(*)
         INTEGER F_RECVTYPE
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Neighbor_alltoallv_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_neighbor_alltoallv_f2vftr_fi

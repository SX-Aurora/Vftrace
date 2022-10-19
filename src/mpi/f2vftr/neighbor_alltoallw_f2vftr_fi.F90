MODULE vftr_mpi_neighbor_alltoallw_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Neighbor_alltoallw_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Neighbor_alltoallw_f2vftr(SENDBUF, F_SENDCOUNTS, &
                                                    F_SDISPLS, F_SENDTYPES, &
                                                    RECVBUF, F_RECVCOUNTS, &
                                                    F_RDISPLS, F_RECVTYPES, &
                                                    F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Neighbor_alltoallw_f2vftr")
         USE mpi, ONLY : MPI_ADDRESS_KIND
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER F_SENDCOUNTS(*)
         INTEGER(KIND=MPI_ADDRESS_KIND) F_SDISPLS(*)
         INTEGER F_SENDTYPES(*)
         INTEGER RECVBUF
         INTEGER F_RECVCOUNTS(*)
         INTEGER(KIND=MPI_ADDRESS_KIND) F_RDISPLS(*)
         INTEGER F_RECVTYPES(*)
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Neighbor_alltoallw_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_neighbor_alltoallw_f2vftr_fi

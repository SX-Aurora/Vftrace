! This file is part of Vftrace.
!
! Vftrace is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 2 of the License, or
! (at your option) any later version.
!
! Vftrace is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

MODULE vftr_mpi_collective_c2F
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Allgather_F, &
             vftr_MPI_Allgatherv_F, &
             vftr_MPI_Allreduce_F, &
             vftr_MPI_Alltoall_F, &
             vftr_MPI_Alltoallv_F, &
             vftr_MPI_Alltoallw_F, &
             vftr_MPI_Bcast_F, &
             vftr_MPI_Gather_F, &
             vftr_MPI_Gatherv_F, &
             vftr_MPI_Reduce_F, &
             vftr_MPI_Reduce_scatter_F, &
             vftr_MPI_Scatter_F, &
             vftr_MPI_Scatterv_F

   INTERFACE 
      SUBROUTINE vftr_MPI_Allgather_F(SENDBUF, SENDCOUNT, F_SENDTYPE, &
                                      RECVBUF, RECVCOUNT, F_RECVTYPE, &
                                      F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Allgather_F")
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER SENDCOUNT
         INTEGER F_SENDTYPE
         INTEGER RECVBUF
         INTEGER RECVCOUNT
         INTEGER F_RECVTYPE
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Allgather_F

      SUBROUTINE vftr_MPI_Allgatherv_F(SENDBUF, SENDCOUNT, F_SENDTYPE, &
                                       RECVBUF, F_RECVCOUNTS, F_DISPLS, &
                                       F_RECVTYPE, F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Allgatherv_F")
         IMPLICIT NONE
         INTEGER :: SENDBUF
         INTEGER :: SENDCOUNT
         INTEGER :: F_SENDTYPE
         INTEGER :: RECVBUF
         INTEGER :: F_RECVCOUNTS(*)
         INTEGER :: F_DISPLS(*)
         INTEGER :: F_RECVTYPE
         INTEGER :: F_COMM
         INTEGER :: F_ERROR
      END SUBROUTINE vftr_MPI_Allgatherv_F

      SUBROUTINE vftr_MPI_Allreduce_F(SENDBUF, RECVBUF, COUNT, &
                                      F_DATATYPE, F_OP, F_COMM, &
                                      F_ERROR) &
         BIND(C, name="vftr_MPI_Allreduce_F")
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER RECVBUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER F_OP
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Allreduce_F

      SUBROUTINE vftr_MPI_Alltoall_F(SENDBUF, SENDCOUNT, F_SENDTYPE, &
                                     RECVBUF, RECVCOUNT, F_RECVTYPE, &
                                     F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Alltoall_F")
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER SENDCOUNT
         INTEGER F_SENDTYPE
         INTEGER RECVBUF
         INTEGER RECVCOUNT
         INTEGER F_RECVTYPE
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Alltoall_F

      SUBROUTINE vftr_MPI_Alltoallv_F(SENDBUF, F_SENDCOUNTS, F_SDISPLS, F_SENDTYPE, &
                                      RECVBUF, F_RECVCOUNTS, F_RDISPLS, F_RECVTYPE, &
                                      F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Alltoallv_F")
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
      END SUBROUTINE vftr_MPI_Alltoallv_F

      SUBROUTINE vftr_MPI_Alltoallw_F(SENDBUF, F_SENDCOUNTS, F_SDISPLS, F_SENDTYPES, &
                                      RECVBUF, F_RECVCOUNTS, F_RDISPLS, F_RECVTYPES, &
                                      F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Alltoallw_F")
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER F_SENDCOUNTS(*)
         INTEGER F_SDISPLS(*)
         INTEGER F_SENDTYPES(*)
         INTEGER RECVBUF
         INTEGER F_RECVCOUNTS(*)
         INTEGER F_RDISPLS(*)
         INTEGER F_RECVTYPES(*)
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Alltoallw_F

      SUBROUTINE vftr_MPI_Bcast_F(BUFFER, COUNT, F_DATATYPE, &
                                  ROOT, F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Bcast_F")
         IMPLICIT NONE
         INTEGER BUFFER
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER ROOT
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Bcast_F

      SUBROUTINE vftr_MPI_Gather_F(SENDBUF, SENDCOUNT, F_SENDTYPE, &
                                   RECVBUF, RECVCOUNT, F_RECVTYPE, &
                                   ROOT, F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Gather_F")
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER SENDCOUNT
         INTEGER F_SENDTYPE
         INTEGER RECVBUF
         INTEGER RECVCOUNT
         INTEGER F_RECVTYPE
         INTEGER ROOT
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Gather_F

      SUBROUTINE vftr_MPI_Gatherv_F(SENDBUF, SENDCOUNT, F_SENDTYPE, &
                                    RECVBUF, F_RECVCOUNTS, F_DISPLS, &
                                    F_RECVTYPE, ROOT, F_COMM, &
                                    F_ERROR) &
         BIND(C, name="vftr_MPI_Gatherv_F")
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER SENDCOUNT
         INTEGER F_SENDTYPE
         INTEGER RECVBUF
         INTEGER F_RECVCOUNTS(*)
         INTEGER F_DISPLS(*)
         INTEGER F_RECVTYPE
         INTEGER ROOT
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Gatherv_F

      SUBROUTINE vftr_MPI_Reduce_F(SENDBUF, RECVBUF, COUNT, F_DATATYPE, &
                                   F_OP, ROOT, F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Reduce_F")
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER RECVBUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER F_OP
         INTEGER ROOT
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Reduce_F

      SUBROUTINE vftr_MPI_Reduce_scatter_F(SENDBUF, RECVBUF, F_RECVCOUNTS, &
                                           F_DATATYPE, F_OP, F_COMM, &
                                           F_ERROR) &
         BIND(C, name="vftr_MPI_Reduce_scatter_F")
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER RECVBUF
         INTEGER F_RECVCOUNTS(*)
         INTEGER F_DATATYPE
         INTEGER F_OP
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Reduce_scatter_F

      SUBROUTINE vftr_MPI_Scatter_F(SENDBUF, SENDCOUNT, F_SENDTYPE, &
                                    RECVBUF, RECVCOUNT, F_RECVTYPE, &
                                    ROOT, F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Scatter_F")
         INTEGER SENDBUF
         INTEGER SENDCOUNT
         INTEGER F_SENDTYPE
         INTEGER RECVBUF
         INTEGER RECVCOUNT
         INTEGER F_RECVTYPE
         INTEGER ROOT
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Scatter_F

      SUBROUTINE vftr_MPI_Scatterv_F(SENDBUF, F_SENDCOUNTS, F_DISPLS, &
                                     F_SENDTYPE, RECVBUF, RECVCOUNT, &
                                     F_RECVTYPE, ROOT, F_COMM, &
                                     F_ERROR) &
         BIND(C, name="vftr_MPI_Scatterv_F")
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER F_SENDCOUNTS(*)
         INTEGER F_DISPLS(*)
         INTEGER F_SENDTYPE
         INTEGER RECVBUF
         INTEGER RECVCOUNT
         INTEGER F_RECVTYPE
         INTEGER ROOT
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Scatterv_F

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_collective_c2F

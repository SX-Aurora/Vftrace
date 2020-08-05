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

#ifdef _MPI

SUBROUTINE MPI_ALLGATHER(SENDBUF, SENDCOUNT, SENDTYPE, &
                         RECVBUF, RECVCOUNT, RECVTYPE, &
                         COMM, ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Allgather_F
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNT
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNT
   INTEGER RECVTYPE
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_MPI_Allgather_F(SENDBUF, SENDCOUNT, SENDTYPE, &
                             RECVBUF, RECVCOUNT, RECVTYPE, &
                             COMM, ERROR)

END SUBROUTINE MPI_ALLGATHER

SUBROUTINE MPI_ALLGATHERV(SENDBUF, SENDCOUNT, SENDTYPE, &
                          RECVBUF, RECVCOUNTS, DISPLS, &
                          RECVTYPE, COMM, ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Allgatherv_F
   IMPLICIT NONE
   INTEGER ::  SENDBUF
   INTEGER ::  SENDCOUNT
   INTEGER ::  SENDTYPE
   INTEGER ::  RECVBUF
   INTEGER ::  RECVCOUNTS(*)
   INTEGER ::  DISPLS(*)
   INTEGER ::  RECVTYPE
   INTEGER ::  COMM
   INTEGER ::  ERROR

   CALL PMPI_Allgatherv(SENDBUF, SENDCOUNT, SENDTYPE, &
                              RECVBUF, RECVCOUNTS, DISPLS, &
                              RECVTYPE, COMM, ERROR)
!TODO: properly call this
!   CALL vftr_MPI_Allgatherv_F(SENDBUF, SENDCOUNT, SENDTYPE, &
!                              RECVBUF, RECVCOUNTS, DISPLS, &
!                              RECVTYPE, COMM, ERROR)

END SUBROUTINE MPI_ALLGATHERV

SUBROUTINE MPI_ALLREDUCE(SENDBUF, RECVBUF, COUNT, &
                         DATATYPE, OP, COMM, &
                         ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Allreduce_F
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER OP
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_MPI_Allreduce_F(SENDBUF, RECVBUF, COUNT, &
                             DATATYPE, OP, COMM, &
                             ERROR)

END SUBROUTINE MPI_ALLREDUCE

SUBROUTINE MPI_ALLTOALL(SENDBUF, SENDCOUNT, SENDTYPE, &
                        RECVBUF, RECVCOUNT, RECVTYPE, &
                        COMM, ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Alltoall_F
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNT
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNT
   INTEGER RECVTYPE
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_MPI_Alltoall_F(SENDBUF, SENDCOUNT, SENDTYPE, &
                            RECVBUF, RECVCOUNT, RECVTYPE, &
                            COMM, ERROR)

END SUBROUTINE MPI_ALLTOALL

SUBROUTINE MPI_ALLTOALLV(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                         RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                         COMM, ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Alltoallv_F
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNTS(*)
   INTEGER SDISPLS(*)
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNTS(*)
   INTEGER RDISPLS(*)
   INTEGER RECVTYPE
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_MPI_Alltoallv_F(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                             RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                             COMM, ERROR)

END SUBROUTINE MPI_ALLTOALLV

SUBROUTINE MPI_ALLTOALLW(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPES, &
                         RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPES, &
                         COMM, ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Alltoallw_F
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNTS(*)
   INTEGER SDISPLS(*)
   INTEGER SENDTYPES(*)
   INTEGER RECVBUF
   INTEGER RECVCOUNTS(*)
   INTEGER RDISPLS(*)
   INTEGER RECVTYPES(*)
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_MPI_Alltoallw_F(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPES, &
                             RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPES, &
                             COMM, ERROR)

END SUBROUTINE MPI_ALLTOALLW

SUBROUTINE MPI_BCAST(BUFFER, COUNT, DATATYPE, &
                     ROOT, COMM, ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Bcast_F
   IMPLICIT NONE
   INTEGER BUFFER
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER ROOT
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_MPI_Bcast_F(BUFFER, COUNT, DATATYPE, &
                         ROOT, COMM, ERROR)

END SUBROUTINE MPI_BCAST

SUBROUTINE MPI_GATHER(SENDBUF, SENDCOUNT, SENDTYPE, &
                      RECVBUF, RECVCOUNT, RECVTYPE, &
                      ROOT, COMM, ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Gather_F
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNT
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNT
   INTEGER RECVTYPE
   INTEGER ROOT
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_MPI_Gather_F(SENDBUF, SENDCOUNT, SENDTYPE, &
                          RECVBUF, RECVCOUNT, RECVTYPE, &
                          ROOT, COMM, ERROR)

END SUBROUTINE MPI_GATHER

SUBROUTINE MPI_GATHERV(SENDBUF, SENDCOUNT, SENDTYPE, &
                       RECVBUF, RECVCOUNTS, DISPLS, &
                       RECVTYPE, ROOT, COMM, &
                       ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Gatherv_F
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNT
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNTS(*)
   INTEGER DISPLS(*)
   INTEGER RECVTYPE
   INTEGER ROOT
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_MPI_Gatherv_F(SENDBUF, SENDCOUNT, SENDTYPE, &
                           RECVBUF, RECVCOUNTS, DISPLS, &
                           RECVTYPE, ROOT, COMM, &
                           ERROR)

END SUBROUTINE MPI_GATHERV

SUBROUTINE MPI_REDUCE(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                      OP, ROOT, COMM, ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Reduce_F
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER COUNT
   INTEGER DATATYPE
   INTEGER OP
   INTEGER ROOT
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_MPI_Reduce_F(SENDBUF, RECVBUF, COUNT, DATATYPE, &
                          OP, ROOT, COMM, ERROR)

END SUBROUTINE MPI_REDUCE

SUBROUTINE MPI_REDUCE_SCATTER(SENDBUF, RECVBUF, RECVCOUNTS, &
                              DATATYPE, OP, COMM, &
                              ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Reduce_scatter_F
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER RECVBUF
   INTEGER RECVCOUNTS(*)
   INTEGER DATATYPE
   INTEGER OP
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_MPI_Reduce_scatter_F(SENDBUF, RECVBUF, RECVCOUNTS, &
                                  DATATYPE, OP, COMM, &
                                  ERROR)

END SUBROUTINE MPI_REDUCE_SCATTER

SUBROUTINE MPI_SCATTER(SENDBUF, SENDCOUNT, SENDTYPE, &
                       RECVBUF, RECVCOUNT, RECVTYPE, &
                       ROOT, COMM, ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Scatter_F
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNT
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNT
   INTEGER RECVTYPE
   INTEGER ROOT
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_MPI_Scatter_F(SENDBUF, SENDCOUNT, SENDTYPE, &
                           RECVBUF, RECVCOUNT, RECVTYPE, &
                           ROOT, COMM, ERROR)

END SUBROUTINE MPI_SCATTER

SUBROUTINE MPI_SCATTERV(SENDBUF, SENDCOUNTS, DISPLS, &
                        SENDTYPE, RECVBUF, RECVCOUNT, &
                        RECVTYPE, ROOT, COMM, &
                        ERROR)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Scatterv_F
   IMPLICIT NONE
   INTEGER SENDBUF
   INTEGER SENDCOUNTS(*)
   INTEGER DISPLS(*)
   INTEGER SENDTYPE
   INTEGER RECVBUF
   INTEGER RECVCOUNT
   INTEGER RECVTYPE
   INTEGER ROOT
   INTEGER COMM
   INTEGER ERROR

   CALL vftr_MPI_Scatterv_F(SENDBUF, SENDCOUNTS, DISPLS, &
                            SENDTYPE, RECVBUF, RECVCOUNT, &
                            RECVTYPE, ROOT, COMM, &
                            ERROR)

END SUBROUTINE MPI_SCATTERV

#endif

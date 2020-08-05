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

SUBROUTINE MPI_ALLGATHERV(SENDBUF, SENDCOUNT, SENDTYPE, &
                          RECVBUF, RECVCOUNTS, DISPLS, &
                          RECVTYPE, COMM, ERROR)
   USE vftr_mpi_allgatherv_c2f, &
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

#endif

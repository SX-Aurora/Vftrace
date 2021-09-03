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

SUBROUTINE MPI_IALLTOALLV(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                          RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                          COMM, REQUEST, ERROR)
   USE vftr_mpi_ialltoallv_f2c_finterface, &
      ONLY : vftr_MPI_Ialltoallv_f2c
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
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
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IALLTOALLV(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                           RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                           COMM, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Ialltoallv_f2c(SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, &
                                   RECVBUF, RECVCOUNTS, RDISPLS, RECVTYPE, &
                                   COMM, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_IALLTOALLV

#endif

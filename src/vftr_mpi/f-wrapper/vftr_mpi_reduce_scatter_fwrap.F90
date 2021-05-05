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

SUBROUTINE MPI_REDUCE_SCATTER(SENDBUF, RECVBUF, RECVCOUNTS, &
                              DATATYPE, OP, COMM, &
                              ERROR)
   USE vftr_mpi_reduce_scatter_f2c, &
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

#endif

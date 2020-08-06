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

SUBROUTINE MPI_Allgather_f08(sendbuf, sendcount, sendtype, &
                             recvbuf, recvcount, recvtype, &
                             comm, error)
   USE vftr_mpi_allgather_c2f, &
      ONLY : vftr_MPI_Allgather_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcount
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcount
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Allgather_F(sendbuf, sendcount, sendtype%MPI_VAL, &
                             recvbuf, recvcount, recvtype%MPI_VAL, &
                             comm%MPI_VAL, tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Allgather_f08

#endif

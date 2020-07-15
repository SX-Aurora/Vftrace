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

SUBROUTINE MPI_Send_f08(buf, count, datatype, dest, tag, comm, ierror)
   USE vftr_mpi_point2point_c2F08, &
      ONLY : vftr_MPI_Send_F08
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm
   IMPLICIT NONE

   TYPE(*), DIMENSION(..), INTENT(IN) :: buf
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: dest
   INTEGER, INTENT(IN) :: tag
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, INTENT(OUT) :: ierror

   CALL vftr_MPI_Send_F08(buf, count, datatype%MPI_VAL, dest, tag, comm%MPI_VAL, ierror)

END SUBROUTINE MPI_Send_f08

SUBROUTINE MPI_Recv_f08(buf, count, datatype, source, tag, comm, status, ierror)
   USE vftr_mpi_point2point_c2F08, &
      ONLY : vftr_MPI_Recv_F08
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm, &
                       MPI_Status
   USE mpi, ONLY : MPI_STATUS_SIZE, &
                       PMPI_Status_f082f, &
                       PMPI_Status_f2f08

   IMPLICIT NONE

   TYPE(*), DIMENSION(..) :: buf
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: source
   INTEGER, INTENT(IN) :: tag
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Status) :: status
   INTEGER, INTENT(OUT) :: ierror

   INTEGER :: tmp_fstatus(MPI_STATUS_SIZE)

   CALL PMPI_Status_f082f(status, tmp_fstatus)
   !ierror = vftr_MPI_Recv_F08(buf, count, datatype%MPI_VAL, source, tag, comm%MPI_VAL, status)
   CALL vftr_MPI_Recv_F08(buf, count, datatype%MPI_VAL, source, tag, comm%MPI_VAL, tmp_fstatus, ierror)
   CALL PMPI_Status_f2f08(tmp_fstatus, status)

END SUBROUTINE MPI_Recv_f08

#endif

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

SUBROUTINE MPI_Iexscan_f08(sendbuf, recvbuf, count, datatype, &
                           op, comm, request, error)
   USE vftr_mpi_iexscan_f082vftr_f08i, &
      ONLY : vftr_MPI_Iexscan_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Scan, &
             MPI_Datatype, &
             MPI_Op, &
             MPI_Comm, &
             MPI_Request
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   TYPE(MPI_Op), INTENT(IN) :: op
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Iexscan(sendbuf, recvbuf, count, datatype, &
                        op, comm, request, tmperror)
   ELSE
      CALL vftr_MPI_Iexscan_f082vftr(sendbuf, recvbuf, count, datatype%MPI_VAL, &
                                     op%MPI_VAL, comm%MPI_VAL, request%MPI_VAL, &
                                     tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Iexscan_f08

#endif

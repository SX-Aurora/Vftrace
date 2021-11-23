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

SUBROUTINE MPI_Sendrecv_replace_f08(buf, count, datatype, dest, sendtag, source, &
                                    recvtag, comm, status, error)
   USE vftr_mpi_sendrecv_replace_f082vftr_f08i, &
      ONLY : vftr_MPI_Sendrecv_replace_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Sendrecv_replace, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Status
   IMPLICIT NONE
   INTEGER buf
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: dest
   INTEGER, INTENT(IN) :: sendtag
   INTEGER, INTENT(IN) :: source
   INTEGER, INTENT(IN) :: recvtag
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Status) :: status
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, &
                                 recvtag, comm, status, tmperror)
   ELSE
      CALL vftr_MPI_Sendrecv_replace_f082vftr(buf, count, datatype%MPI_VAL, dest, sendtag, source, &
                                              recvtag, comm%MPI_VAL, status, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror


END SUBROUTINE MPI_Sendrecv_replace_f08

#endif

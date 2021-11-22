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

SUBROUTINE MPI_Startall_f08(count, array_of_requests, error)
   USE vftr_mpi_startall_f082vftr_f08i, &
      ONLY : vftr_MPI_Startall_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Startall, &
             MPI_Request
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Request), INTENT(INOUT) :: array_of_requests(count)
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror
   INTEGER, DIMENSION(count) :: tmparray_of_requests
   INTEGER :: i

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Startall(count, array_of_requests, tmperror)
   ELSE
      DO i = 1, count
         tmparray_of_requests(i) = array_of_requests(i)%MPI_VAL
      END DO
      CALL vftr_MPI_Startall_f082vftr(count, tmparray_of_requests, tmperror)
      DO i = 1, count
         array_of_requests(i)%MPI_VAL = tmparray_of_requests(i)
      END DO
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Startall_f08

#endif 


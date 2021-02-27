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

SUBROUTINE MPI_Init_f08(ierror)
   USE vftr_after_mpi_init_f082c, &
      ONLY : vftr_after_mpi_init_F08
   USE mpi_f08, ONLY : PMPI_Init

   IMPLICIT NONE

   INTEGER, OPTIONAL, INTENT(OUT) :: ierror

   IF (PRESENT(ierror)) THEN
      CALL PMPI_Init(ierror)
   ELSE
      CALL PMPI_Init()
   END IF

   CALL vftr_after_mpi_init_F08()

END SUBROUTINE MPI_Init_f08

#endif

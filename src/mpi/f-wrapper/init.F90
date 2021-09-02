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

SUBROUTINE MPI_INIT(IERROR)
   USE vftr_after_mpi_init_f2c_finterface, &
      ONLY : vftr_after_mpi_init_f2c
   USE mpi, &
      ONLY : PMPI_INIT
   IMPLICIT NONE
   INTEGER, INTENT(OUT) :: IERROR

   CALL PMPI_INIT(IERROR)

   CALL vftr_after_mpi_init_f2c()

END SUBROUTINE MPI_INIT

#endif

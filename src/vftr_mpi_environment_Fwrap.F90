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
   USE vftr_mpi_environment_c2F, &
      ONLY : vftr_after_mpi_init
   USE mpi, ONLY : PMPI_INIT

   IMPLICIT NONE

   INTEGER, INTENT(OUT) :: IERROR

   CALL PMPI_INIT(IERROR)

   CALL vftr_after_mpi_init()

END SUBROUTINE MPI_INIT

SUBROUTINE MPI_FINALIZE(IERROR)
   USE vftr_mpi_environment_c2F, &
      ONLY : vftr_finalize
   USE mpi, ONLY : PMPI_FINALIZE

   IMPLICIT NONE

   INTEGER, INTENT(OUT) :: IERROR

   ! it is neccessary to finalize vftrace here, in order to properly communicat stack ids
   ! between processes. After MPI_Finalize communication between processes is prohibited
   CALL vftr_finalize()

   CALL PMPI_FINALIZE(IERROR)

END SUBROUTINE MPI_FINALIZE

SUBROUTINE MPI_PCONTROL(LEVEL)
   USE vftr_mpi_environment_c2F, &
      ONLY : vftr_MPI_Pcontrol_F
   
   IMPLICIT NONE

   INTEGER, INTENT(IN) :: LEVEL

   CALL vftr_Pcontrol_F()

END SUBROUTINE MPI_PCONTROL

#endif

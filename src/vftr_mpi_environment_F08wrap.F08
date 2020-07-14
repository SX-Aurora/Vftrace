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
   USE vftr_mpi_environment_c2F08, &
      ONLY : vftr_after_mpi_init
   USE mpi_f08, ONLY : PMPI_Init

   IMPLICIT NONE

   INTEGER, OPTIONAL, INTENT(OUT) :: ierror

   IF (PRESENT(ierror)) THEN
      CALL PMPI_Init(ierror)
   ELSE
      CALL PMPI_Init()
   END IF

   CALL vftr_after_mpi_init()

END SUBROUTINE MPI_INIT_f08

SUBROUTINE MPI_Finalize_f08(ierror)
   USE vftr_mpi_environment_c2F08, &
      ONLY : vftr_finalize
   USE mpi_f08, ONLY : PMPI_Finalize

   IMPLICIT NONE

   INTEGER, OPTIONAL, INTENT(OUT) :: ierror

   ! it is neccessary to finalize vftrace here, in order to properly communicat stack ids
   ! between processes. After MPI_Finalize communication between processes is prohibited
   CALL vftr_finalize()

   CALL PMPI_Finalize(ierror)

END SUBROUTINE MPI_Finalize_f08

SUBROUTINE MPI_Pcontrol_f08(level)
   USE vftr_mpi_environment_c2F08, &
      ONLY : vftr_MPI_Pcontrol_F08
   
   IMPLICIT NONE

   INTEGER, INTENT(IN) :: level

   CALL vftr_MPI_Pcontrol_F08(level)

END SUBROUTINE MPI_Pcontrol_f08

#endif

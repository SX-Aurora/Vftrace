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

SUBROUTINE MPI_Get_f08(origin_addr, origin_count, origin_datatype, &
                       target_rank, target_disp, target_count, &
                       target_datatype, win, error)
   USE vftr_mpi_get_f082c_f08interface, &
      ONLY : vftr_MPI_Get_f082c
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Get_f08, &
             MPI_Datatype, &
             MPI_Win, &
             MPI_ADDRESS_KIND
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: origin_addr
   INTEGER, INTENT(IN) :: origin_count
   TYPE(MPI_Datatype), INTENT(IN) :: origin_datatype
   INTEGER target_rank
   INTEGER(KIND=MPI_ADDRESS_KIND) target_disp
   INTEGER target_count
   TYPE(MPI_Datatype), INTENT(IN) :: target_datatype
   TYPE(MPI_Win), INTENT(IN) :: win
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Get_f08(origin_addr, origin_count, origin_datatype, &
                        target_rank, target_disp, target_count, &
                        target_datatype, win, tmperror)
   ELSE
      CALL vftr_MPI_Get_f082c(origin_addr, origin_count, origin_datatype%MPI_VAL, &
                              target_rank, target_disp, target_count, &
                              target_datatype%MPI_VAL, win%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Get_f08

#endif

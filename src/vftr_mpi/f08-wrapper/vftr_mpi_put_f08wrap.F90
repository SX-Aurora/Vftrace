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

SUBROUTINE MPI_Put_f08(origin_addr, origin_count, origin_datatype, &
                       target_rank, target_disp, target_count, &
                       target_datatype, win, error)
   USE vftr_mpi_put_f2c, &
      ONLY : vftr_MPI_Put_F
   USE mpi_f08, ONLY: MPI_Datatype, &
                      MPI_Win, &
                      MPI_ADDRESS_KIND
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: origin_addr
   INTEGER, INTENT(IN) :: origin_count
   TYPE(MPI_Datatype), INTENT(IN) :: origin_datatype
   INTEGER, INTENT(IN) :: target_rank
   INTEGER(KIND=MPI_ADDRESS_KIND) target_disp
   INTEGER, INTENT(IN) :: target_count
   TYPE(MPI_Datatype), INTENT(IN) :: target_datatype
   TYPE(MPI_Win), INTENT(IN) :: win
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Put_F(origin_addr, origin_count, origin_datatype%MPI_VAL, &
                       target_rank, target_disp, target_count, &
                       target_datatype%MPI_VAL, win%MPI_VAL, tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Put_f08

#endif

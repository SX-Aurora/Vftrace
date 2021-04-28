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

SUBROUTINE MPI_Compare_and_swap_f08(origin_addr, compare_addr, result_addr, &
                                    datatype, target_rank, target_disp, &
                                    win, error)
   USE vftr_mpi_compare_and_swap_f082c_f08interface, &
      ONLY : vftr_MPI_Compare_and_swap_f082c
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Win, &
                       MPI_ADDRESS_KIND
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: origin_addr
   INTEGER, INTENT(IN) :: compare_addr
   INTEGER, INTENT(IN) :: result_addr
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER target_rank
   INTEGER(KIND=MPI_ADDRESS_KIND) target_disp
   TYPE(MPI_Win), INTENT(IN) :: win
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Compare_and_swap_f082c(origin_addr, compare_addr, result_addr, &
                                    datatype%MPI_VAL, target_rank, target_disp, &
                                    win%MPI_VAL, tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Compare_and_swap_f08

#endif

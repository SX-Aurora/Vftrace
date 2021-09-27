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

SUBROUTINE MPI_Fetch_and_op_f08(origin_addr, result_addr, datatype, &
                                target_rank, target_disp, op, win, &
                                error)
   USE vftr_mpi_fetch_and_op_f082vftr_f08i, &
      ONLY : vftr_MPI_Fetch_and_op_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Fetch_and_op, &
             MPI_Datatype, &
             MPI_Op, &
             MPI_Win, &
             MPI_ADDRESS_KIND
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: origin_addr
   INTEGER, INTENT(IN) :: result_addr
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER target_rank
   INTEGER(KIND=MPI_ADDRESS_KIND) target_disp
   TYPE(MPI_Op), INTENT(IN) :: op
   TYPE(MPI_Win), INTENT(IN) :: win
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Fetch_and_op(origin_addr, result_addr, datatype, &
                             target_rank, target_disp, op, win, &
                             tmperror)
   ELSE
      CALL vftr_MPI_Fetch_and_op_f082vftr(origin_addr, result_addr, datatype%MPI_VAL, &
                                       target_rank, target_disp, op%MPI_VAL, win%MPI_VAL, &
                                       tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Fetch_and_op_f08

#endif

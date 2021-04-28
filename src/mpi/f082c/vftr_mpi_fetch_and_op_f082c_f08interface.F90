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

MODULE vftr_mpi_fetch_and_op_f082c_f08interface
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Fetch_and_op_f082c

   INTERFACE

      SUBROUTINE vftr_MPI_Fetch_and_op_f082c(origin_addr, result_addr, f_datatype, &
                                             target_rank, target_disp, f_op, f_win, &
                                             f_error) &
         BIND(C, name="vftr_MPI_Fetch_and_op_f082c")
         USE mpi_f08, ONLY : MPI_ADDRESS_KIND
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: origin_addr
         INTEGER, INTENT(IN) :: result_addr
         INTEGER, INTENT(IN) :: f_datatype
         INTEGER, INTENT(IN) :: target_rank
         INTEGER(KIND=MPI_ADDRESS_KIND), INTENT(IN) :: target_disp
         INTEGER, INTENT(IN) :: f_win
         INTEGER, INTENT(IN) :: f_op
         INTEGER, INTENT(OUT) :: f_error
      END SUBROUTINE vftr_MPI_Fetch_and_op_f082c

   END INTERFACE

#endif 

CONTAINS

END MODULE vftr_mpi_fetch_and_op_f082c_f08interface

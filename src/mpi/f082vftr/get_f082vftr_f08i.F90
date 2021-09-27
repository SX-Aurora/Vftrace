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

MODULE vftr_mpi_get_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Get_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Get_f082vftr(origin_addr, origin_count, f_origin_datatype, &
                                       target_rank, target_disp, target_count, &
                                       f_target_datatype, f_win, f_error) &
         BIND(C, name="vftr_MPI_Get_f082vftr")
         USE mpi_f08, ONLY : MPI_ADDRESS_KIND
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: origin_addr
         INTEGER, INTENT(IN) :: origin_count
         INTEGER, INTENT(IN) :: f_origin_datatype
         INTEGER, INTENT(IN) :: target_rank
         INTEGER(KIND=MPI_ADDRESS_KIND), INTENT(IN) :: target_disp
         INTEGER, INTENT(IN) :: target_count
         INTEGER, INTENT(IN) :: f_target_datatype
         INTEGER, INTENT(IN) :: f_win
         INTEGER, INTENT(OUT) :: f_error
      END SUBROUTINE vftr_MPI_Get_f082vftr

   END INTERFACE

#endif 

CONTAINS

END MODULE vftr_mpi_get_f082vftr_f08i
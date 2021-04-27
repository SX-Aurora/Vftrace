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

MODULE vftr_finalize_f082c
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_finalize_F08

   INTERFACE

      SUBROUTINE vftr_finalize_F08(do_normalize_stacks) &
         BIND(c, NAME="vftr_finalize")
         USE ISO_C_BINDING, ONLY : c_bool
         IMPLICIT NONE
         LOGICAL(KIND=c_bool) :: do_normalize_stacks
      END SUBROUTINE vftr_finalize_F08

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_finalize_f082c

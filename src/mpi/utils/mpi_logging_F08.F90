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

MODULE vftr_mpi_logging_F08

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_no_mpi_logging_F08

   INTERFACE

      FUNCTION vftr_no_mpi_logging_int_F08() &
         BIND(c, NAME="vftr_no_mpi_logging_int")
         USE ISO_C_BINDING, ONLY : C_INT
         IMPLICIT NONE
         INTEGER(KIND=C_INT) :: vftr_no_mpi_logging_int_F08
      END FUNCTION vftr_no_mpi_logging_int_F08

   END INTERFACE

CONTAINS

   LOGICAL FUNCTION vftr_no_mpi_logging_F08()
      vftr_no_mpi_logging_F08 = vftr_no_mpi_logging_int_F08() == 1
      RETURN
   END FUNCTION vftr_no_mpi_logging_F08

END MODULE vftr_mpi_logging_F08

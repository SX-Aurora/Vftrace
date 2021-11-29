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

MODULE vftr_mpi_request_free_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Request_free_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Request_free_f082vftr(f_request, f_error) &
         BIND(C, name="vftr_MPI_Request_free_f082vftr")
         IMPLICIT NONE
         INTEGER, INTENT(INOUT) :: F_REQUEST
         INTEGER, INTENT(OUT) :: F_ERROR
      END SUBROUTINE vftr_MPI_Request_free_f082vftr

   END INTERFACE

#endif 

CONTAINS

END MODULE vftr_mpi_request_free_f082vftr_f08i
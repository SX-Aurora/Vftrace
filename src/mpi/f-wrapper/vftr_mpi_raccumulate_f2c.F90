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

MODULE vftr_mpi_raccumulate_f2c
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Raccumulate_F

   INTERFACE

      SUBROUTINE vftr_MPI_Raccumulate_F(ORIGIN_ADDR, ORIGIN_COUNT, F_ORIGIN_DATATYPE, &
                                        TARGET_RANK, TARGET_DISP, TARGET_COUNT, &
                                        F_TARGET_DATATYPE, F_OP, F_WIN, F_REQUEST, &
                                        F_ERROR) &
         BIND(C, name="vftr_MPI_Raccumulate_F")
         USE mpi, ONLY: MPI_ADDRESS_KIND
         IMPLICIT NONE
         INTEGER ORIGIN_ADDR
         INTEGER ORIGIN_COUNT
         INTEGER F_ORIGIN_DATATYPE
         INTEGER TARGET_RANK
         INTEGER(KIND=MPI_ADDRESS_KIND) TARGET_DISP
         INTEGER TARGET_COUNT
         INTEGER F_TARGET_DATATYPE
         INTEGER F_OP
         INTEGER F_WIN
         INTEGER F_REQUEST
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Raccumulate_F

   END INTERFACE

#endif 

CONTAINS

END MODULE vftr_mpi_raccumulate_f2c

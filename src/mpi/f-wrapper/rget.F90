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

SUBROUTINE MPI_RGET(ORIGIN_ADDR, ORIGIN_COUNT, ORIGIN_DATATYPE, &
                    TARGET_RANK, TARGET_DISP, TARGET_COUNT, &
                    TARGET_DATATYPE, WIN, REQUEST, ERROR)
   USE vftr_mpi_rget_f2vftr_fi, &
      ONLY : vftr_MPI_Rget_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_RGET, &
             MPI_ADDRESS_KIND
   IMPLICIT NONE
   INTEGER ORIGIN_ADDR
   INTEGER ORIGIN_COUNT
   INTEGER ORIGIN_DATATYPE
   INTEGER TARGET_RANK
   INTEGER(KIND=MPI_ADDRESS_KIND) TARGET_DISP
   INTEGER TARGET_COUNT
   INTEGER TARGET_DATATYPE
   INTEGER WIN
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_RGET(ORIGIN_ADDR, ORIGIN_COUNT, ORIGIN_DATATYPE, &
                     TARGET_RANK, TARGET_DISP, TARGET_COUNT, &
                     TARGET_DATATYPE, WIN, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Rget_f2vftr(ORIGIN_ADDR, ORIGIN_COUNT, ORIGIN_DATATYPE, &
                             TARGET_RANK, TARGET_DISP, TARGET_COUNT, &
                             TARGET_DATATYPE, WIN, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_RGET

#endif

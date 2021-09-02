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

SUBROUTINE MPI_RACCUMULATE(ORIGIN_ADDR, ORIGIN_COUNT, ORIGIN_DATATYPE, &
                           TARGET_RANK, TARGET_DISP, TARGET_COUNT, &
                           TARGET_DATATYPE, OP, WIN, REQUEST, ERROR)
   USE vftr_mpi_raccumulate_f2c_finterface, &
      ONLY : vftr_MPI_Raccumulate_f2c
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY: PMPI_RACCUMULATE, &
            MPI_ADDRESS_KIND
   IMPLICIT NONE
   INTEGER ORIGIN_ADDR
   INTEGER ORIGIN_COUNT
   INTEGER ORIGIN_DATATYPE
   INTEGER TARGET_RANK
   INTEGER(KIND=MPI_ADDRESS_KIND) TARGET_DISP
   INTEGER TARGET_COUNT
   INTEGER TARGET_DATATYPE
   INTEGER OP
   INTEGER WIN
   INTEGER REQUEST
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_RACCUMULATE(ORIGIN_ADDR, ORIGIN_COUNT, ORIGIN_DATATYPE, &
                            TARGET_RANK, TARGET_DISP, TARGET_COUNT, &
                            TARGET_DATATYPE, OP, WIN, REQUEST, ERROR)
   ELSE
      CALL vftr_MPI_Raccumulate_f2c(ORIGIN_ADDR, ORIGIN_COUNT, ORIGIN_DATATYPE, &
                                    TARGET_RANK, TARGET_DISP, TARGET_COUNT, &
                                    TARGET_DATATYPE, OP, WIN, REQUEST, ERROR)
   END IF

END SUBROUTINE MPI_RACCUMULATE

#endif

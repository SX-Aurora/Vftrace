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

SUBROUTINE MPI_TESTSOME(INCOUNT, ARRAY_OREQUESTS, OUTCOUNT, &
                        ARRAY_OINDICES, ARRAY_OSTATUSES, ERROR)
   USE vftr_mpi_testsome_f2c_finterface, &
      ONLY : vftr_MPI_Testsome_f2c
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY: PMPI_TESTSOME, &
            MPI_STATUS_SIZE
   IMPLICIT NONE
   INTEGER INCOUNT
   INTEGER ARRAY_OREQUESTS(*)
   INTEGER OUTCOUNT
   INTEGER ARRAY_OINDICES(*)
   INTEGER ARRAY_OSTATUSES(MPI_STATUS_SIZE,*)
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_TESTSOME(INCOUNT, ARRAY_OREQUESTS, OUTCOUNT, &
                         ARRAY_OINDICES, ARRAY_OSTATUSES, ERROR)
   ELSE
      CALL vftr_MPI_Testsome_f2c(INCOUNT, ARRAY_OREQUESTS, OUTCOUNT, &
                                 ARRAY_OINDICES, ARRAY_OSTATUSES, ERROR)
   END IF

END SUBROUTINE MPI_TESTSOME

#endif 

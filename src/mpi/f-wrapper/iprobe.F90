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

SUBROUTINE MPI_IPROBE(SOURCE, TAG, COMM, FLAG, STATUS, ERROR)
   USE vftr_mpi_iprobe_f2c_finterface, &
      ONLY : vftr_MPI_Iprobe_f2c
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_IPROBE, &
             MPI_STATUS_SIZE
   IMPLICIT NONE
   INTEGER SOURCE
   INTEGER TAG
   INTEGER COMM
   LOGICAL FLAG
   INTEGER STATUS(MPI_STATUS_SIZE)
   INTEGER ERROR

   INTEGER TMPFLAG

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_IPROBE(SOURCE, TAG, COMM, FLAG, STATUS, ERROR)
   ELSE
      CALL vftr_MPI_Iprobe_f2c(SOURCE, TAG, COMM, TMPFLAG, STATUS, ERROR)
      FLAG = (TMPFLAG /= 0)
   END IF

END SUBROUTINE MPI_IPROBE

#endif 

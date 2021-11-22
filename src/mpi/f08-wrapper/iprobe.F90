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

SUBROUTINE MPI_Iprobe_f08(source, tag, comm, flag, status, error)
   USE vftr_mpi_iprobe_f082vftr_f08i, &
      ONLY : vftr_MPI_Iprobe_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Iprobe, &
             MPI_Comm, &
             MPI_Status
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: source
   INTEGER, INTENT(IN) :: tag
   TYPE(MPI_Comm), INTENT(IN) :: comm
   LOGICAL, INTENT(OUT) :: flag
   TYPE(MPI_Status) :: status
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmpflag, tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Iprobe(source, tag, comm, flag, status, tmperror)
   ELSE
      CALL vftr_MPI_Iprobe_f082vftr(source, tag, comm%MPI_VAL, tmpflag, status, tmperror)
      FLAG = (TMPFLAG /= 0)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Iprobe_f08

#endif 

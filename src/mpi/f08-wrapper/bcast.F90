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

SUBROUTINE MPI_Bcast_f08(buffer, count, datatype, &
                         root, comm, error)
   USE vftr_mpi_bcast_f082vftr_f08i, &
      ONLY : vftr_MPI_Bcast_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Bcast_f08, &
             MPI_Datatype, &
             MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: buffer
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: root
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Bcast_f08(buffer, count, datatype, &
                          root, comm, tmperror)
   ELSE
      CALL vftr_MPI_Bcast_f082vftr(buffer, count, datatype%MPI_VAL, &
                                root, comm%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Bcast_f08

#endif

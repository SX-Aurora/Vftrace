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

SUBROUTINE MPI_Ialltoallw_f08(sendbuf, sendcounts, sdispls, sendtypes, &
                              recvbuf, recvcounts, rdispls, recvtypes, &
                              comm, request, error)
   USE vftr_mpi_ialltoallw_f082vftr_f08i, &
      ONLY : vftr_MPI_Ialltoallw_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Ialltoallw, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Request
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcounts(*)
   INTEGER, INTENT(IN) :: sdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: sendtypes(*)
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcounts(*)
   INTEGER, INTENT(IN) :: rdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: recvtypes(*)
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror
   INTEGER, DIMENSION(:), ALLOCATABLE :: tmpsendtypes
   INTEGER, DIMENSION(:), ALLOCATABLE :: tmprecvtypes
   INTEGER :: comm_size, i
   LOGICAL :: isintercom

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes, &
                           recvbuf, recvcounts, rdispls, recvtypes, &
                           comm, request, tmperror)
   ELSE
      CALL PMPI_Comm_test_inter(comm, isintercom, tmperror)
      IF (isintercom) THEN
         CALL PMPI_Comm_remote_size(comm, comm_size, tmperror)
      ELSE
         CALL PMPI_Comm_size(comm, comm_size, tmperror)
      END IF
   
      ALLOCATE(tmpsendtypes(comm_size))
      ALLOCATE(tmprecvtypes(comm_size))
      DO i = 1, comm_size
         tmpsendtypes(i) = sendtypes(i)%MPI_VAL
      END DO
      DO i = 1, comm_size
         tmprecvtypes(i) = recvtypes(i)%MPI_VAL
      END DO
   
      CALL vftr_MPI_Ialltoallw_f082vftr(sendbuf, sendcounts, sdispls, tmpsendtypes, &
                                        recvbuf, recvcounts, rdispls, tmprecvtypes, &
                                        comm%MPI_VAL, request%MPI_VAL, tmperror)
   
      DEALLOCATE(tmpsendtypes)
      DEALLOCATE(tmprecvtypes)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Ialltoallw_f08

#endif
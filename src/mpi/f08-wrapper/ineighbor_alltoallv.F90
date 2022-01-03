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

SUBROUTINE MPI_Ineighbor_alltoallv_f08(sendbuf, sendcounts, sdispls, sendtype, &
                                       recvbuf, recvcounts, rdispls, recvtype, &
                                       comm, request, error)
   USE vftr_mpi_ineighbor_alltoallv_f082vftr_f08i, &
      ONLY : vftr_MPI_Ineighbor_alltoallv_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Ineighbor_alltoallv, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Request
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcounts(*)
   INTEGER, INTENT(IN) :: sdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcounts(*)
   INTEGER, INTENT(IN) :: rdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Ineighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype, &
                                    recvbuf, recvcounts, rdispls, recvtype, &
                                    comm, request, tmperror)
   ELSE
      CALL vftr_MPI_Ineighbor_alltoallv_f082vftr(sendbuf, sendcounts, &
                                                 sdispls, sendtype%MPI_VAL, &
                                                 recvbuf, recvcounts, &
                                                 rdispls, recvtype%MPI_VAL, &
                                                 comm%MPI_VAL, request%MPI_VAL, &
                                                 tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Ineighbor_alltoallv_f08

#endif

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

SUBROUTINE MPI_Ineighbor_alltoallw_f08(sendbuf, sendcounts, sdispls, sendtypes, &
                                       recvbuf, recvcounts, rdispls, recvtypes, &
                                       comm, request, error)
   USE vftr_mpi_ineighbor_alltoallw_f082vftr_f08i, &
      ONLY : vftr_MPI_Ineighbor_alltoallw_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Ineighbor_alltoallw, &
             PMPI_Topo_test, &
             PMPI_Comm_rank, &
             PMPI_Graph_neighbors_count, &
             PMPI_Cartdim_get, &
             PMPI_Dist_graph_neighbors_count, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Request, &
             MPI_ADDRESS_KIND, &
             MPI_GRAPH, MPI_CART, MPI_DIST_GRAPH
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcounts(*)
   INTEGER(KIND=MPI_ADDRESS_KIND), INTENT(IN) :: sdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: sendtypes(*)
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcounts(*)
   INTEGER(KIND=MPI_ADDRESS_KIND), INTENT(IN) :: rdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: recvtypes(*)
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror
   INTEGER, DIMENSION(:), ALLOCATABLE :: tmpsendtypes
   INTEGER, DIMENSION(:), ALLOCATABLE :: tmprecvtypes
   INTEGER :: sizein, sizeout, i, rank
   INTEGER :: topology
   LOGICAL :: weighted

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Ineighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes, &
                                    recvbuf, recvcounts, rdispls, recvtypes, &
                                    comm, request, tmperror)
   ELSE
      CALL PMPI_Topo_test(comm, topology, tmperror)
      SELECT CASE(topology)
         CASE(MPI_GRAPH)
            CALL PMPI_Comm_rank(comm, rank, tmperror)
            CALL PMPI_Graph_neighbors_count(comm, rank, sizein);
            sizeout = sizein
         CASE(MPI_CART)
            CALL PMPI_Cartdim_get(comm, sizein)
            ! Number of neighbors for cartesian communicators is always 2*ndims
            sizein = 2*sizein
            sizeout = sizein
         CASE(MPI_DIST_GRAPH)
            CALL PMPI_Dist_graph_neighbors_count(comm, sizein, sizeout, weighted)
      END SELECT

      ALLOCATE(tmpsendtypes(sizeout))
      ALLOCATE(tmprecvtypes(sizein))
      DO i = 1, sizeout
         tmpsendtypes(i) = sendtypes(i)%MPI_VAL
      END DO
      DO i = 1, sizein
         tmprecvtypes(i) = recvtypes(i)%MPI_VAL
      END DO

      CALL vftr_MPI_Ineighbor_alltoallw_f082vftr(sendbuf, sendcounts, &
                                                 sdispls, tmpsendtypes, &
                                                 recvbuf, recvcounts, &
                                                 rdispls, tmprecvtypes, &
                                                 comm%MPI_VAL, request%MPI_VAL, &
                                                 tmperror)

      DEALLOCATE(tmpsendtypes)
      DEALLOCATE(tmprecvtypes)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Ineighbor_alltoallw_f08

#endif

#ifdef _MPI

SUBROUTINE MPI_Neighbor_alltoallw_f08(sendbuf, sendcounts, sdispls, sendtypes, &
                                      recvbuf, recvcounts, rdispls, recvtypes, &
                                      comm, error)
   USE vftr_mpi_neighbor_alltoallw_f082vftr_f08i, &
      ONLY : vftr_MPI_Neighbor_alltoallw_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE vftr_sync_time_F08, &
      ONLY : vftr_estimate_sync_time
   USE mpi_f08, &
      ONLY : PMPI_Neighbor_alltoallw, &
             PMPI_Topo_test, &
             PMPI_Comm_rank, &
             PMPI_Graph_neighbors_count, &
             PMPI_Cartdim_get, &
             PMPI_Dist_graph_neighbors_count, &
             MPI_Datatype, &
             MPI_Comm, &
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
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror
   INTEGER, DIMENSION(:), ALLOCATABLE :: tmpsendtypes
   INTEGER, DIMENSION(:), ALLOCATABLE :: tmprecvtypes
   INTEGER :: sizein, sizeout, i, rank
   INTEGER :: topology
   LOGICAL :: weighted

   CALL vftr_estimate_sync_time("MPI_Neighbor_alltoallw_sync", comm)

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Neighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes, &
                                   recvbuf, recvcounts, rdispls, recvtypes, &
                                   comm, tmperror)
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

      CALL vftr_MPI_Neighbor_alltoallw_f082vftr(sendbuf, sendcounts, &
                                                sdispls, tmpsendtypes, &
                                                recvbuf, recvcounts, &
                                                rdispls, tmprecvtypes, &
                                                comm%MPI_VAL, tmperror)

      DEALLOCATE(tmpsendtypes)
      DEALLOCATE(tmprecvtypes)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Neighbor_alltoallw_f08

#endif

PROGRAM ineighbor_alltoallv_dist_graph

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: sendcounts
   INTEGER, DIMENSION(:), ALLOCATABLE :: sdispls
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: recvcounts
   INTEGER, DIMENSION(:), ALLOCATABLE :: rdispls
   INTEGER :: nstot, nrtot


   INTEGER, PARAMETER :: nnodes = 1
   INTEGER, DIMENSION(nnodes) :: sources
   INTEGER, DIMENSION(nnodes) :: degrees
   INTEGER, DIMENSION(:), ALLOCATABLE :: destinations
   LOGICAL, PARAMETER :: reorder = .FALSE.
   INTEGER :: indegree, outdegree
   LOGICAL :: weighted
   TYPE(MPI_Comm) :: comm_dist_graph
   INTEGER, DIMENSION(:), ALLOCATABLE :: inneighbors
   INTEGER, DIMENSION(:), ALLOCATABLE :: inweights
   INTEGER, DIMENSION(:), ALLOCATABLE :: outneighbors
   INTEGER, DIMENSION(:), ALLOCATABLE :: outweights
   INTEGER :: i, ineighbor

   LOGICAL :: valid_data
   INTEGER :: refval

   TYPE(MPI_Request) :: myrequest
   TYPE(MPI_Status) :: mystatus

   INTEGER :: ierr

   CALL MPI_Init(ierr)

   ! Get the number of processes
   CALL MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr);
   ! Get rank of processes
   CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr);

   ! require cmd-line argument
   IF (COMMAND_ARGUMENT_COUNT() < 1) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./ineighbor_alltoallv_dist_graph <msgsize in integers>"
      STOP 1
   END IF

   ! requires precicely 4 processes
   IF (comm_size /= 4) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "requires precicely 4 processes. Start with -np 4!"
      STOP 1
   ENDIF

   ! create the cartesian communicator
   sources(1) = my_rank
   SELECT CASE(my_rank)
      CASE(0)
         degrees(1) = 3
         ALLOCATE(destinations(degrees(1)))
         destinations(:) = [1,1,2]
      CASE(1)
         degrees(1) = 3
         ALLOCATE(destinations(degrees(1)))
         destinations(:) = [0,0,2]
      CASE(2)
         degrees(1) = 2
         ALLOCATE(destinations(degrees(1)))
         destinations(:) = [0,3]
      CASE(3)
         degrees(1) = 1
         ALLOCATE(destinations(degrees(1)))
         destinations(:) = [3]
   END SELECT
   CALL MPI_Dist_graph_create(MPI_COMM_WORLD, nnodes, sources, &
                              degrees, destinations, &
                              MPI_UNWEIGHTED, MPI_INFO_NULL, &
                              reorder, comm_dist_graph, ierr)
   DEALLOCATE(destinations)
   CALL MPI_Dist_graph_neighbors_count(comm_dist_graph, indegree, outdegree, weighted, ierr)
   ALLOCATE(inneighbors(indegree))
   ALLOCATE(inweights(indegree))
   ALLOCATE(outneighbors(outdegree))
   ALLOCATE(outweights(outdegree))
   CALL MPI_Dist_graph_neighbors(comm_dist_graph, &
                                 indegree, inneighbors, inweights, &
                                 outdegree, outneighbors, outweights, &
                                 ierr)

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   nints = nints + my_rank
   ALLOCATE(sendcounts(outdegree))
   ALLOCATE(sdispls(outdegree))
   nstot = 0
   DO ineighbor = 0, outdegree - 1
      sendcounts(ineighbor+1) = nints
      sdispls(ineighbor+1) = nstot
      nstot = nstot + sendcounts(ineighbor+1)
   END DO
   ALLOCATE(sbuffer(nstot))
   sbuffer(:) = my_rank
   ALLOCATE(recvcounts(indegree))
   ALLOCATE(rdispls(indegree))
   nrtot = 0
   DO ineighbor = 0, indegree - 1
      recvcounts(ineighbor+1) = nints - my_rank + inneighbors(ineighbor+1)
      rdispls(ineighbor+1) = nrtot
      nrtot = nrtot + recvcounts(ineighbor+1)
   END DO
   ALLOCATE(rbuffer(nrtot))
   rbuffer(:) = -1

   ! Message cycle
   CALL MPI_Ineighbor_alltoallv(sbuffer, sendcounts, sdispls, MPI_INTEGER, &
                                rbuffer, recvcounts, rdispls, MPI_INTEGER, &
                                comm_dist_graph, myrequest, ierr)
   CALL MPI_Wait(myrequest, mystatus, ierr)

   ! validate data
   valid_data = .TRUE.
   DO ineighbor = 0, indegree - 1
      refval = inneighbors(ineighbor+1)
      DO i = 1, recvcounts(ineighbor+1)
         IF (rbuffer(i+rdispls(ineighbor+1)) /= refval) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") &
               "Rank ", my_rank, " received faulty data from rank ", inneighbors(ineighbor+1)
            valid_data = .FALSE.
            EXIT
         END IF
      END DO
   END DO

   DEALLOCATE(inneighbors)
   DEALLOCATE(inweights)
   DEALLOCATE(outneighbors)
   DEALLOCATE(outweights)
   DEALLOCATE(rbuffer)
   DEALLOCATE(recvcounts)
   DEALLOCATE(rdispls)
   DEALLOCATE(sbuffer)
   DEALLOCATE(sendcounts)
   DEALLOCATE(sdispls)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM ineighbor_alltoallv_dist_graph

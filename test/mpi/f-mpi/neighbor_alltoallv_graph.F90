PROGRAM neighbor_alltoallv_graph

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

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

   INTEGER, PARAMETER :: nnodes = 4
   INTEGER, PARAMETER :: nedges = 11
   INTEGER, DIMENSION(nnodes) :: index = [3,6,9,11]
   INTEGER, DIMENSION(nedges) :: edges = [1,1,2,0,0,2,0,1,3,2,3]
   LOGICAL, PARAMETER :: reorder = .FALSE.
   INTEGER :: comm_graph
   INTEGER :: nneighbors
   INTEGER, DIMENSION(:), ALLOCATABLE :: neighbors
   INTEGER :: i, ineighbor

   LOGICAL :: valid_data
   INTEGER :: refval

   INTEGER :: ierr

   CALL MPI_Init(ierr)

   ! Get the number of processes
   CALL MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr);
   ! Get rank of processes
   CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr);

   ! require cmd-line argument
   IF (COMMAND_ARGUMENT_COUNT() < 1) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./neighbor_alltoallv_graph <msgsize in integers>"
      STOP 1
   END IF

   ! requires precicely 4 processes
   IF (comm_size /= 4) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "requires precicely 4 processes. Start with -np 4!"
      STOP 1
   ENDIF

   ! create the cartesian communicator
   CALL MPI_Graph_create(MPI_COMM_WORLD, nnodes, index, edges, reorder, comm_graph, ierr)
   
   CALL MPI_Graph_neighbors_count(comm_graph, my_rank, nneighbors, ierr)
   ALLOCATE(neighbors(nneighbors))
   CALL MPI_Graph_neighbors(comm_graph, my_rank, nneighbors, neighbors, ierr)

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   nints = nints + my_rank
   ALLOCATE(sendcounts(nneighbors))
   ALLOCATE(sdispls(nneighbors))
   nstot = 0
   DO ineighbor = 0, nneighbors - 1
      sendcounts(ineighbor+1) = nints
      sdispls(ineighbor+1) = nstot
      nstot = nstot + sendcounts(ineighbor+1)
   END DO
   ALLOCATE(sbuffer(nstot))
   sbuffer(:) = my_rank
   ALLOCATE(recvcounts(nneighbors))
   ALLOCATE(rdispls(nneighbors))
   nrtot = 0
   DO ineighbor = 0, nneighbors - 1
      recvcounts(ineighbor+1) = nints - my_rank + neighbors(ineighbor+1)
      rdispls(ineighbor+1) = nrtot
      nrtot = nrtot + recvcounts(ineighbor+1)
   END DO
   ALLOCATE(rbuffer(nrtot))
   rbuffer(:) = -1

   ! Message cycle
   CALL MPI_Neighbor_alltoallv(sbuffer, sendcounts, sdispls, MPI_INTEGER, &
                               rbuffer, recvcounts, rdispls, MPI_INTEGER, &
                               comm_graph, ierr)

   ! validate data
   valid_data = .TRUE.
   DO ineighbor = 0, nneighbors - 1
      refval = neighbors(ineighbor+1)
      DO i = 1, recvcounts(ineighbor+1)
         IF (rbuffer(i+rdispls(ineighbor+1)) /= refval) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") &
               "Rank ", my_rank, " received faulty data from rank ", neighbors(ineighbor+1)
            valid_data = .FALSE.
            EXIT
         END IF
      END DO
   END DO

   DEALLOCATE(neighbors)
   DEALLOCATE(rbuffer)
   DEALLOCATE(recvcounts)
   DEALLOCATE(rdispls)
   DEALLOCATE(sbuffer)
   DEALLOCATE(sendcounts)
   DEALLOCATE(sdispls)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM neighbor_alltoallv_graph

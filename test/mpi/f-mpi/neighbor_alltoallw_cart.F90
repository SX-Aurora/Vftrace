PROGRAM neighbor_alltoallw_cart

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: sendcounts
   INTEGER(KIND=MPI_ADDRESS_KIND), DIMENSION(:), ALLOCATABLE :: sdispls
   INTEGER, DIMENSION(:), ALLOCATABLE :: sendtypes
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: recvcounts
   INTEGER(KIND=MPI_ADDRESS_KIND), DIMENSION(:), ALLOCATABLE :: rdispls
   INTEGER, DIMENSION(:), ALLOCATABLE :: recvtypes
   INTEGER :: nstot, nrtot
   INTEGER :: dummyint

   INTEGER, PARAMETER :: ndims = 3
   INTEGER, DIMENSION(ndims) :: dims = [2,2,1]
   LOGICAL, DIMENSION(ndims) :: periods = [.FALSE., .FALSE., .FALSE.]
   LOGICAL, PARAMETER :: reorder = .FALSE.
   INTEGER :: comm_cart
   INTEGER, PARAMETER :: nneighbors = 2*ndims
   INTEGER, DIMENSION(nneighbors) :: neighbors
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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./neighbor_alltoallw_cart <msgsize in integers>"
      STOP 1
   END IF

   ! requires precicely 4 processes
   IF (comm_size /= 4) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "requires precicely 4 processes. Start with -np 4!"
      STOP 1
   ENDIF

   ! create the cartesian communicator
   CALL MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, comm_cart, ierr)
   ! fill the neighbor list
   neighbors(:) = -1
   SELECT CASE (my_rank)
      CASE (0)
         neighbors(2) = 2
         neighbors(4) = 1
      CASE (1)
         neighbors(2) = 3
         neighbors(3) = 0
      CASE (2)
         neighbors(1) = 0
         neighbors(4) = 3
      CASE (3)
         neighbors(1) = 1
         neighbors(3) = 2
   END SELECT

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   nints = nints + my_rank
   ALLOCATE(sendcounts(nneighbors))
   ALLOCATE(sdispls(nneighbors))
   ALLOCATE(sendtypes(nneighbors))
   nstot = 0
   DO ineighbor = 0, nneighbors - 1
      IF (neighbors(ineighbor+1) /= -1)  THEN
         sendcounts(ineighbor+1) = nints
      ELSE
         sendcounts(ineighbor+1) = 0
      END IF
      sdispls(ineighbor+1) = INT(nstot*C_SIZEOF(dummyint), MPI_ADDRESS_KIND)
      sendtypes(ineighbor+1) = MPI_INTEGER
      nstot = nstot + sendcounts(ineighbor+1)
   END DO
   ALLOCATE(sbuffer(nstot))
   sbuffer(:) = my_rank
   ALLOCATE(recvcounts(nneighbors))
   ALLOCATE(rdispls(nneighbors))
   ALLOCATE(recvtypes(nneighbors))
   nrtot = 0
   DO ineighbor = 0, nneighbors - 1
      IF (neighbors(ineighbor+1) /= -1)  THEN
         recvcounts(ineighbor+1) = nints - my_rank + neighbors(ineighbor+1)
      ELSE
         recvcounts(ineighbor+1) = 0
      END IF
      rdispls(ineighbor+1) = INT(nrtot*C_SIZEOF(dummyint), MPI_ADDRESS_KIND)
      recvtypes(ineighbor+1) = MPI_INTEGER
      nrtot = nrtot + recvcounts(ineighbor+1)
   END DO
   ALLOCATE(rbuffer(nrtot))
   rbuffer(:) = -1

DO i = 0, comm_size - 1
IF (my_rank == i) THEN
WRITE(*,*) "Rank: ", my_rank
WRITE(*,*) "   neighbors: ", neighbors
WRITE(*,*) "   sendcount: ", sendcounts
WRITE(*,*) "   sdispls:   ", sdispls
WRITE(*,*) "   sendtypes: ", ALL(sendtypes == MPI_INTEGER)
WRITE(*,*) "   sendbuff:  ", sbuffer
WRITE(*,*) "   recvcount: ", recvcounts
WRITE(*,*) "   rdispls:   ", rdispls
WRITE(*,*) "   recvtypes: ", ALL(recvtypes == MPI_INTEGER)
WRITE(*,*) "   recvbuff:  ", rbuffer
WRITE(*,*) ""
ENDIF
CALL MPI_BARRIER(MPI_COMM_WORLD, ierr)
END DO
   ! Message cycle
   CALL MPI_Neighbor_alltoallw(sbuffer, sendcounts, sdispls, sendtypes, &
                               rbuffer, recvcounts, rdispls, recvtypes, &
                               comm_cart, ierr)

   ! validate data
   valid_data = .TRUE.
!   DO ineighbor = 0, nneighbors - 1
!      refval = neighbors(ineighbor+1)
!      DO i = 1, recvcounts(ineighbor+1)
!         IF (rbuffer(i+rdispls(ineighbor+1)/C_SIZEOF(dummyint)) /= refval) THEN
!            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") &
!               "Rank ", my_rank, " received faulty data from rank ", neighbors(ineighbor+1)
!            valid_data = .FALSE.
!            EXIT
!         END IF
!      END DO
!   END DO
!
!   DEALLOCATE(rbuffer)
!   DEALLOCATE(recvcounts)
!   DEALLOCATE(rdispls)
!   DEALLOCATE(recvtypes)
!   DEALLOCATE(sbuffer)
!   DEALLOCATE(sendcounts)
!   DEALLOCATE(sdispls)
!   DEALLOCATE(sendtypes)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM neighbor_alltoallw_cart

PROGRAM neighbor_allgather_cart

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:,:), ALLOCATABLE :: rbuffer

   INTEGER, PARAMETER :: ndims = 3
   INTEGER, DIMENSION(ndims) :: dims = [2,2,1]
   LOGICAL, DIMENSION(ndims) :: periods = [.FALSE., .FALSE., .FALSE.]
   LOGICAL, PARAMETER :: reorder = .FALSE.
   INTEGER :: comm_cart
   INTEGER, PARAMETER :: nneighbors = 2*ndims
   INTEGER, DIMENSION(nneighbors) :: neighbors
   INTEGER :: ineighbor

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./neighbor_allgather_cart <msgsize in integers>"
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
   ALLOCATE(sbuffer(nints))
   sbuffer(:) = my_rank
   ALLOCATE(rbuffer(nints,nneighbors))
   rbuffer(:,:) = -1

   ! Message cycle
   CALL MPI_Neighbor_allgather(sbuffer, nints, MPI_INTEGER, &
                               rbuffer, nints, MPI_INTEGER, &
                               comm_cart, ierr)

   ! validate data
   valid_data = .TRUE.
   DO ineighbor = 0, nneighbors - 1
      refval = neighbors(ineighbor+1)
      IF (ANY(rbuffer(:,ineighbor+1) /= refval)) THEN
         WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
            "Rank ", my_rank, " received faulty data from rank ", neighbors(ineighbor+1)
         valid_data = .FALSE.
      END IF
   END DO

   DEALLOCATE(rbuffer)
   DEALLOCATE(sbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM neighbor_allgather_cart

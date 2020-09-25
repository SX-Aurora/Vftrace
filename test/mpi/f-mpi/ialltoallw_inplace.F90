PROGRAM ialltoallw_inplace

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER :: nrtot
   INTEGER :: dummyint
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: scounts
   INTEGER, DIMENSION(:), ALLOCATABLE :: sdispls
   INTEGER, DIMENSION(:), ALLOCATABLE :: stypes
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: rcounts
   INTEGER, DIMENSION(:), ALLOCATABLE :: rdispls
   INTEGER, DIMENSION(:), ALLOCATABLE :: rtypes

   INTEGER, PARAMETER :: rootrank = 0
   INTEGER :: irank, i

   LOGICAL :: valid_data

   INTEGER :: myrequest
   INTEGER :: mystatus(MPI_STATUS_SIZE)

   INTEGER :: ierr

   CALL MPI_Init(ierr)

   ! Get the number of processes
   CALL MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr);
   ! Get rank of processes
   CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr);

   ! Write information
   IF (comm_size < 2) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "At least two ranks are required"
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "RUN again with '-np 2'"
      CALL FLUSH(OUTPUT_UNIT)
      CALL MPI_Finalize(ierr)
      STOP
   END IF

   ! require cmd-line argument
   IF (COMMAND_ARGUMENT_COUNT() < 1) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./ialltoallw_inplace <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints

   ! prepare special arrays for send
   ALLOCATE(scounts(0))
   ALLOCATE(sdispls(0))
   ALLOCATE(stypes(0))

   ! prepare special arrays for recv
   ALLOCATE(rcounts(comm_size))
   ALLOCATE(rdispls(comm_size))
   ALLOCATE(rtypes(comm_size))
   nrtot = 0
   DO irank = 0, comm_size - 1
      rcounts(irank+1) = nints
      rdispls(irank+1) = nrtot*C_SIZEOF(dummyint)
      rtypes(irank+1) = MPI_INTEGER
      nrtot = nrtot + rcounts(irank+1)
   END DO
   ALLOCATE(rbuffer(nrtot))
   rbuffer(:) = my_rank

   ! Messaging
   CALL MPI_Ialltoallw(MPI_IN_PLACE, scounts, sdispls, stypes, &
                       rbuffer, rcounts, rdispls, rtypes, &
                       MPI_COMM_WORLD, myrequest, ierr)
   CALL MPI_Wait(myrequest, mystatus, ierr)

   WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") &
      "Communicating with all ranks"

   ! validate data
   valid_data = .TRUE.
   DO irank = 0, comm_size - 1
      DO i = 1, rcounts(irank+1)
         IF (rbuffer(i+rdispls(irank+1)/C_SIZEOF(dummyint)) /= irank) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") &
               "Rank ", my_rank, " received faulty data from rank ", irank
            valid_data = .FALSE.
            EXIT
         END IF
      END DO
   END DO
   DEALLOCATE(rbuffer)

   DEALLOCATE(rcounts)
   DEALLOCATE(rdispls)
   DEALLOCATE(rtypes)

   DEALLOCATE(scounts)
   DEALLOCATE(sdispls)
   DEALLOCATE(stypes)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM ialltoallw_inplace

PROGRAM raccumulate

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank
   
   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: srbuffer

   INTEGER :: window
   INTEGER(KIND=MPI_ADDRESS_KIND) :: winsize

   INTEGER :: targetrank
   INTEGER(KIND=MPI_ADDRESS_KIND) :: targetdisp

   LOGICAL :: valid_data

   INTEGER, DIMENSION(:,:), ALLOCATABLE :: mystatuses
   INTEGER, DIMENSION(:), ALLOCATABLE :: myrequests

   INTEGER :: ierr

   CALL MPI_Init(ierr)

   ! Get the number of processes
   CALL MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr);
   ! Get rank of processes
   CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr);

   ! Write information
   IF (comm_size < 2) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I6,A)") "At least two ranks are required"
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I6,A)") "RUN again with '-np 2'"
      CALL FLUSH(OUTPUT_UNIT)
      CALL MPI_Finalize(ierr)
      STOP
   END IF

   ! require cmd-line argument
   IF (COMMAND_ARGUMENT_COUNT() < 1) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./raccumulate <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(srbuffer(nints))
   srbuffer(:) = my_rank

   ! open memory to remote memory access
   winsize = nints*(BIT_SIZE(srbuffer(1))/8)
   CALL MPI_Win_create(srbuffer, winsize, 1, MPI_INFO_NULL, MPI_COMM_WORLD, window, ierr)
   CALL MPI_Win_fence(0, window, ierr)

   ! Remote memory access
   IF (my_rank == 0) THEN
      srbuffer(:) = comm_size
      ! send to every other rank
      ALLOCATE(myrequests(comm_size-1))
      targetdisp = 0
      DO targetrank = 1, comm_size - 1
         WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") "Putting data remotely on rank ", targetrank
         CALL MPI_Raccumulate(srbuffer, nints, MPI_INTEGER, targetrank, targetdisp, &
                              nints, MPI_INTEGER, MPI_SUM, window, myrequests(targetrank), &
                              ierr)
      END DO
      ALLOCATE(mystatuses(MPI_STATUS_SIZE,comm_size-1))
      CALL MPI_Waitall(comm_size-1, myrequests, mystatuses, ierr)
      DEALLOCATE(myrequests)
      DEALLOCATE(mystatuses)
   END IF

   CALL MPI_Win_fence(0, window, ierr)
   CALL MPI_Win_free(window, ierr)

   CALL MPI_Barrier(MPI_COMM_WORLD, ierr)

   ! validate data
   valid_data = .TRUE.
   IF (ANY(srbuffer(:) /= comm_size+my_rank)) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") &
         "Rank ", my_rank, " received faulty data from rank", 0
      valid_data = .FALSE.
   END IF

   DEALLOCATE(srbuffer)

   CALL MPI_Finalize(ierr)
   
   IF (.NOT.valid_data) STOP 1
END PROGRAM raccumulate

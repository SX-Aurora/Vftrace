PROGRAM get

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank
   
   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:,:), ALLOCATABLE :: rbuffer

   TYPE(MPI_WIN) :: window
   INTEGER(KIND=MPI_ADDRESS_KIND) :: winsize

   INTEGER :: targetrank
   INTEGER(KIND=MPI_ADDRESS_KIND) :: targetdisp

   LOGICAL :: valid_data

   INTEGER :: ipeer

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./get <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(sbuffer(nints))
   sbuffer(:) = my_rank
   ALLOCATE(rbuffer(nints,comm_size-1))
   rbuffer(:,:) = -1

   ! open memory to remote memory access
   winsize = nints*(BIT_SIZE(sbuffer(1))/8)
   CALL MPI_Win_create(sbuffer, winsize, 1, MPI_INFO_NULL, MPI_COMM_WORLD, window, ierr)
   CALL MPI_Win_fence(0, window, ierr)

   ! Remote memory access
   IF (my_rank == 0) THEN
      ! send to every other rank
      targetdisp = 0
      DO targetrank = 1, comm_size - 1
         WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") "Collecting data remotely from rank ", targetrank
         CALL MPI_Get(rbuffer(:,targetrank), nints, MPI_INTEGER, targetrank, targetdisp, &
                      nints, MPI_INTEGER, window, ierr)
      END DO
   END IF

   CALL MPI_Win_fence(0, window, ierr)
   CALL MPI_Win_free(window, ierr)

   CALL MPI_Barrier(MPI_COMM_WORLD, ierr)

   ! validate data
   valid_data = .TRUE.
   IF (my_rank == 0) THEN
      DO ipeer = 1, comm_size-1
         IF (ANY(rbuffer(:,ipeer) /= ipeer)) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") &
               "Rank ", my_rank, " received faulty data from rank", ipeer
            valid_data = .FALSE.
         END IF
      END DO
   END IF

   DEALLOCATE(sbuffer)
   DEALLOCATE(rbuffer)

   CALL MPI_Finalize(ierr)
   
   IF (.NOT.valid_data) STOP 1
END PROGRAM get

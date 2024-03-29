PROGRAM compare_and_swap

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: originbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: resultbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: comparebuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: targetbuffer

   TYPE(MPI_Win) :: window
   INTEGER(KIND=MPI_ADDRESS_KIND) :: winsize

   INTEGER(KIND=MPI_ADDRESS_KIND) :: targetdisp

   LOGICAL :: valid_data

   INTEGER :: i, irank, refresult

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./compare_and_swap <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   IF (my_rank == 0) THEN
      ALLOCATE(originbuffer(nints))
      originbuffer(:) = my_rank
      ALLOCATE(resultbuffer(nints))
      resultbuffer(:) = 0
      ALLOCATE(comparebuffer(nints))
      DO i = 1, nints
         comparebuffer(i) = 2*(MOD(i-1, comm_size)/2)
      END DO
      ALLOCATE(targetbuffer(nints))
   ELSE
      ALLOCATE(originbuffer(0))
      ALLOCATE(resultbuffer(0))
      ALLOCATE(comparebuffer(0))
      ALLOCATE(targetbuffer(nints))
      targetbuffer(:) = my_rank
   END IF

   ! open memory to remote memory access
   winsize = nints*(BIT_SIZE(targetbuffer(1))/8)
   CALL MPI_Win_create(targetbuffer, winsize, 1, MPI_INFO_NULL, MPI_COMM_WORLD, window, ierr)
   CALL MPI_Win_fence(0, window, ierr)

   ! Remote memory access
   IF (my_rank == 0) THEN
      ! send to every other rank
      targetdisp = 0
      DO irank = 1, comm_size - 1
         ! Origin stays unchanged
         ! Resultbuffer gets a copy of the target buffer from remote process
         ! The remote target buffer gets the sum of origin+itself
         CALL MPI_Compare_and_swap(originbuffer(irank+1), comparebuffer(irank+1), &
                                   resultbuffer(irank+1), MPI_INTEGER, &
                                   irank, targetdisp, window, ierr)
      END DO
   END IF

   CALL MPI_Win_fence(0, window, ierr)
   CALL MPI_Win_free(window, ierr)

   CALL MPI_Barrier(MPI_COMM_WORLD, ierr)

   ! validate data
   valid_data = .TRUE.
   IF (my_rank == 0) THEN
      ! contents of origin buffer should be 0
      refresult = 0
      DO i = 1, nints
         IF (originbuffer(i) /= refresult) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
               "Rank ", my_rank, " received faulty data"
            valid_data = .FALSE.
            EXIT
         END IF
      END DO
      ! contents of result buffer should be the largest rank
      DO i = 1, nints
         IF (i <= comm_size) THEN
            refresult = i-1
         ELSE
            refresult = 0
         END IF
         IF (resultbuffer(i) /= refresult) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
               "Rank ", my_rank, " received faulty data"
            valid_data = .FALSE.
            EXIT
         END IF
      END DO
      ! contents of comparebuffer should be unchanged
      DO i = 1, nints
         refresult = 2*(MOD(i-1, comm_size)/2)
         IF (comparebuffer(i) /= refresult) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
               "Rank ", my_rank, " received faulty data"
            valid_data = .FALSE.
            EXIT
         END IF
      END DO
   ELSE
      ! contents of target buffer should be my_rank except the first element
      DO i = 1, nints
         IF (i == 1 .AND. MOD(my_rank, 2) == 0) THEN
            refresult = 0
         ELSE
            refresult = my_rank
         END IF
         IF (targetbuffer(i) /= refresult) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
               "Rank ", my_rank, " received faulty data"
            valid_data = .FALSE.
            EXIT
         END IF
      END DO
   END IF

   DEALLOCATE(originbuffer)
   DEALLOCATE(resultbuffer)
   DEALLOCATE(comparebuffer)
   DEALLOCATE(targetbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM compare_and_swap

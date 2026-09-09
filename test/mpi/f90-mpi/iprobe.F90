PROGRAM iprobe

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer

   INTEGER :: recvstatus(MPI_STATUS_SIZE)

   LOGICAL :: flag

   LOGICAL :: valid_data

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./iprobe <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(sbuffer(nints))
   sbuffer(:) = my_rank
   ALLOCATE(rbuffer(nints))
   rbuffer(:) = -1

   ! Message cycle
   valid_data = .TRUE.
   IF (my_rank == 0) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Sending message from rank ", my_rank, " to rank", 1
      CALL MPI_Send(sbuffer, nints, MPI_INTEGER, 1, 0, MPI_COMM_WORLD, ierr)
   ELSE
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Receiving message on rank ", my_rank, " from rank", 0
      CALL MPI_Iprobe(0, 0, MPI_COMM_WORLD, flag, recvstatus, ierr)
      CALL MPI_Recv(rbuffer, nints, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, recvstatus, ierr)
      ! validate data
      IF (ANY(rbuffer /= 0)) THEN
         WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Rank ", my_rank, " received faulty data from rank", 0
         valid_data = .FALSE.
      END IF
   END IF

   DEALLOCATE(sbuffer)
   DEALLOCATE(rbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM iprobe

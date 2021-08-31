PROGRAM ibcast

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: buffer

   INTEGER, PARAMETER :: sendrank = 0

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./ibcast <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(buffer(nints))
   buffer(:) = my_rank

   ! Message
   CALL MPI_Ibcast(buffer, nints, MPI_INTEGER, sendrank, MPI_COMM_WORLD, myrequest, ierr)
   CALL MPI_Wait(myrequest, mystatus, ierr)

   IF (my_rank == sendrank) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") &
         "Broadcasted message from rank ", my_rank
   END IF

   ! validate data
   valid_data = .TRUE.
   IF (ANY(buffer /= sendrank)) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") &
         "Rank ", my_rank, " received faulty data from rank ", sendrank
      valid_data = .FALSE.
   END IF

   DEALLOCATE(buffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM ibcast

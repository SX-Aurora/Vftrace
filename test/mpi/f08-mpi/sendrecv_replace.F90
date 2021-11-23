PROGRAM sendrecv_replace

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: srbuffer

   TYPE(MPI_Status) :: recvstatus

   INTEGER :: destrank, sourcerank

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./send_recv <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(srbuffer(nints))
   srbuffer(:) = my_rank

   ! Messaging
   destrank = MOD((my_rank+1),comm_size)
   sourcerank = my_rank - 1
   IF (sourcerank < 0) THEN
      sourcerank = comm_size - 1
   END IF

   CALL MPI_Sendrecv_replace(srbuffer, nints, MPI_INTEGER, &
                             destrank, 0, &
                             sourcerank, 0, &
                             MPI_COMM_WORLD, recvstatus, ierr)

   valid_data = .TRUE.
   IF (ANY(srbuffer /= sourcerank)) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Rank ", my_rank, " received faulty data from rank", sourcerank
      valid_data = .FALSE.
   END IF

   DEALLOCATE(srbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM sendrecv_replace

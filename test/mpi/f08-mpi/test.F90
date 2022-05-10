PROGRAM test

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: srbuffer

   TYPE(MPI_Status) :: mystatus
   TYPE(MPI_Request) :: request

   LOGICAL :: flag

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./test <msgsize in intergers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(srbuffer(nints))
   srbuffer(:) = my_rank


   IF (my_rank == 0) THEN
      ! send to every other rank
      CALL MPI_Isend(srbuffer, nints, MPI_INTEGER, 1, 0, MPI_COMM_WORLD, request, ierr)
      flag = .FALSE.
      DO WHILE (.NOT. flag)
         Call MPI_Test(request, flag, mystatus, ierr)
         IF (flag) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "Sending request is completed"
         ELSE
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "Sending request is not completed"
            CALL SLEEP(1)
         END IF
      END DO
   ELSE IF (my_rank == 1) THEN
      CALL SLEEP(2)
      CALL MPI_Recv(srbuffer, nints, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, mystatus, ierr)
   END IF
   CALL MPI_Barrier(MPI_COMM_WORLD, ierr);

   DEALLOCATE(srbuffer)

   CALL MPI_Finalize(ierr)

END PROGRAM test

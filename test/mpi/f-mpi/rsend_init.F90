PROGRAM rsend_init

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER :: nruns = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer

   INTEGER :: mystat(MPI_STATUS_SIZE)
   INTEGER, DIMENSION(:), ALLOCATABLE :: myrequest
   INTEGER :: ireq, irun

   INTEGER :: recvrank

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
   IF (COMMAND_ARGUMENT_COUNT() < 2) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./rsend_init <msgsize in integers> <nrepetitions>"
      STOP 1
   END IF

   CALL GET_COMMAND_ARGUMENT(2,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nruns

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(sbuffer(nints))
   sbuffer(:) = my_rank
   ALLOCATE(rbuffer(nints))
   rbuffer(:) = -1
   ALLOCATE(myrequest(comm_size-1))

   ! Message cycle
   valid_data = .TRUE.
   IF (my_rank == 0) THEN
      ! prepare send to every other rank
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") "Initialize sending message from rank ", my_rank
      DO recvrank = 1, comm_size-1
         CALL MPI_Rsend_init(sbuffer, nints, MPI_INTEGER, recvrank, 0, MPI_COMM_WORLD, &
                            myrequest(recvrank), ierr)
      END DO

      DO irun = 1, nruns
         ! send to every other rank
         WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") "Sending messages from rank ", my_rank
         DO ireq = 1, comm_size-1
            CALL MPI_Start(myrequest(ireq), ierr)
         END DO
         ! wait for completion of non-blocking sends
         DO ireq = 1, comm_size-1
            CALL MPI_Wait(myrequest(ireq), mystat, ierr)
         END DO
      END DO
      ! mark persistent requests for deallocation
      DO ireq = 1, comm_size-1
         CALL MPI_Request_free(myrequest(ireq), ierr);
      END DO
   ELSE 
      DO irun = 1, nruns
         WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Receiving message from rank", 0
         CALL MPI_Recv(rbuffer, nints, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, mystat, ierr)
         ! validate data
         IF (ANY(rbuffer /= 0)) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Rank ", my_rank, " received faulty data from rank", 0
            valid_data = .FALSE.
         END IF
      END DO
   END IF

   DEALLOCATE(sbuffer)
   DEALLOCATE(rbuffer)

   DEALLOCATE(myrequest)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM rsend_init

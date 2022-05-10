PROGRAM send_irecv

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:,:), ALLOCATABLE :: rbuffer

   TYPE(MPI_Status) :: mystat
   TYPE(MPI_Request), DIMENSION(:), ALLOCATABLE :: myrequest
   INTEGER :: ireq

   INTEGER :: sendrank, recvrank

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./send_irecv <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(sbuffer(nints))
   sbuffer(:) = my_rank
   ALLOCATE(rbuffer(nints,comm_size))
   rbuffer(:,:) = -1
   ALLOCATE(myrequest(comm_size))

   ! prepare non-blocking receive
   DO sendrank = 0, comm_size-1
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") "Preparing message receiving from rank ", sendrank
      CALL MPI_Irecv(rbuffer(:,sendrank+1), nints, MPI_INTEGER, sendrank, &
                     sendrank, MPI_COMM_WORLD, myrequest(sendrank+1), ierr)
   END DO

   ! sending
   DO recvrank = 0, comm_size-1
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") "Sending message from rank ", my_rank
      CALL MPI_Send(sbuffer, nints, MPI_INTEGER, recvrank, &
                    my_rank, MPI_COMM_WORLD, ierr)
   END DO



   ! Wait for completion of non-blocking receives
   DO ireq = 1, comm_size
      CALL MPI_Wait(myrequest(ireq), mystat, ierr)
   END DO

   ! verify communication data
   valid_data = .TRUE.
   ! validate data
   DO sendrank = 0, comm_size - 1
      IF (ANY(rbuffer(:,sendrank+1) /= sendrank)) THEN
          WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Rank ", my_rank, " received faulty data from rank", sendrank
          valid_data = .FALSE.
      END IF
   END DO

   DEALLOCATE(sbuffer)
   DEALLOCATE(rbuffer)

   DEALLOCATE(myrequest)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM send_irecv

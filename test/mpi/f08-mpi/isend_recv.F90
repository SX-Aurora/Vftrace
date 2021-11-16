PROGRAM send_recv

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer

   TYPE(MPI_Status) :: mystat
   TYPE(MPI_Request), DIMENSION(:), ALLOCATABLE :: myrequest
   INTEGER :: reqidx, ireq

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./send_recv <msgsize in integers>"
      STOP 1
   END IF

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
   DO sendrank = 0, comm_size-1
      IF (my_rank == sendrank) THEN
         ! send to every other rank
         reqidx = 1
         DO recvrank = 0, comm_size-1
            IF (my_rank /= recvrank) THEN
               WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Sending message from rank ", my_rank, " to rank", recvrank
               CALL MPI_Isend(sbuffer, nints, MPI_INTEGER, recvrank, 0, MPI_COMM_WORLD, myrequest(reqidx), ierr)
               reqidx = reqidx + 1
            END IF
         END DO
         ! Wait for completion of non-blocking sends
         DO ireq = 1, comm_size-1
            CALL MPI_Wait(myrequest(ireq), mystat, ierr)
         END DO
      ELSE 
         WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Receiving message on rank ", my_rank, " from rank", sendrank
         CALL MPI_Recv(rbuffer, nints, MPI_INTEGER, sendrank, 0, MPI_COMM_WORLD, mystat, ierr)
         ! validate data
         IF (ANY(rbuffer /= sendrank)) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Rank ", my_rank, " received faulty data from rank", sendrank
            valid_data = .FALSE.
         END IF
      END IF
   END DO

   DEALLOCATE(sbuffer)
   DEALLOCATE(rbuffer)

   DEALLOCATE(myrequest)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM send_recv

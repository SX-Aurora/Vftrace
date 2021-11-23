PROGRAM waitany

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank
   
   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: srbuffer

   TYPE(MPI_Status) :: mystat
   TYPE(MPI_Request), DIMENSION(:), ALLOCATABLE :: requests
   INTEGER :: idx, ireq

   INTEGER :: recvrank

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./testany <msgsize in intergers>"
      STOP 1
   END IF

   ! allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(srbuffer(nints))
   srbuffer(:) = my_rank

   IF (my_rank == 0) THEN
      ALLOCATE(requests(comm_size-1))
      ! send to every other rank
      DO recvrank = 1, comm_size - 1
         CALL MPI_Isend(srbuffer, nints, MPI_INTEGER, recvrank, 0, MPI_COMM_WORLD, requests(recvrank), ierr)
      END DO
      CALL MPI_Waitany(comm_size-1, requests, idx, mystat, ierr)
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I6,A)") &
         "Sending request to rank ", idx, " is completed"
      DO ireq = 1, comm_size - 1
         IF (ireq /= idx) THEN
            CALL MPI_Wait(requests(ireq), mystat, ierr)
         END IF
      END DO
      DEALLOCATE(requests)
   ELSE
      CALL SLEEP(2*(MOD(my_rank,2)))
      CALL MPI_Recv(srbuffer, nints, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, mystat, ierr)
   END IF

   DEALLOCATE(srbuffer)

   CALL MPI_Finalize(ierr)
   
END PROGRAM waitany

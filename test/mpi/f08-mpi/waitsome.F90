PROGRAM waitsome

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: srbuffer

   TYPE(MPI_Status) :: mystat
   TYPE(MPI_Status), DIMENSION(:), ALLOCATABLE :: statuses
   TYPE(MPI_Request), DIMENSION(:), ALLOCATABLE :: requests
   INTEGER, DIMENSION(:), ALLOCATABLE :: idxs
   INTEGER :: iidx, ireq
   INTEGER :: outcount

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./waitsome <msgsize in intergers>"
      STOP 1
   END IF

   ! allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(srbuffer(nints))
   srbuffer(:) = my_rank

   IF (my_rank == 0) THEN
      ALLOCATE(requests(comm_size-1))
      ALLOCATE(statuses(comm_size-1))
      ! send to every other rank
      DO recvrank = 1, comm_size - 1
         CALL MPI_Isend(srbuffer, nints, MPI_INTEGER, recvrank, 0, MPI_COMM_WORLD, requests(recvrank), ierr)
      END DO
      CALL SLEEP(1)
      outcount = 0
      ALLOCATE(idxs(comm_size-1))
      CALL MPI_Waitsome(comm_size-1, requests, outcount, idxs, statuses, ierr)
      DO iidx = 1, outcount
         WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I6,A)") &
            "Sending request to rank ", idxs(iidx), " is completed"
      END DO
      DO ireq = 1, comm_size-1
         IF (requests(ireq) /= MPI_REQUEST_NULL) THEN
            CALL MPI_Wait(requests(ireq), statuses(1), ierr)
         END IF
      END DO
      DEALLOCATE(requests)
      DEALLOCATE(statuses)
   ELSE
      CALL SLEEP(2*(MOD(my_rank,2)))
      CALL MPI_Recv(srbuffer, nints, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, mystat, ierr)
   END IF

   DEALLOCATE(srbuffer)

   CALL MPI_Finalize(ierr)

END PROGRAM waitsome

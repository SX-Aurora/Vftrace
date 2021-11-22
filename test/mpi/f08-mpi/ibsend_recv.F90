PROGRAM ibsend_recv

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER :: bufsize = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer
   INTEGER, DIMENSION(:), POINTER :: buffer
   TYPE(C_PTR) :: buffer_c_addr

   TYPE(MPI_Status) :: recvstatus

   INTEGER :: sendrank

   LOGICAL :: valid_data

   TYPE(MPI_Request) :: myrequest
   TYPE(MPI_Status) :: mystatus

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./ibsend_recv <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(sbuffer(nints))
   sbuffer(:) = my_rank
   ALLOCATE(rbuffer(nints))
   rbuffer(:) = -1

   ! allocating and attatching MPI-buffered mode buffer
   NULLIFY(buffer)
   ALLOCATE(buffer(nints+MPI_BSEND_OVERHEAD))
   buffer_c_addr = C_LOC(buffer)
   bufsize = INT(nints*C_SIZEOF(buffer(1))+MPI_BSEND_OVERHEAD)
   CALL MPI_Buffer_attach(buffer, bufsize, ierr)

   ! Message cycle
   valid_data = .TRUE.
   IF (my_rank == 0) THEN
      ! recv from every other rank
      DO sendrank = 1, comm_size-1
         WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Receiving message on rank ", my_rank, " from rank", sendrank
         CALL MPI_Recv(rbuffer, nints, MPI_INTEGER, sendrank, sendrank, MPI_COMM_WORLD, recvstatus, ierr)
         ! validate data
         IF (ANY(rbuffer /= sendrank)) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Rank ", my_rank, " received faulty data from rank", sendrank
            valid_data = .FALSE.
         END IF
      END DO
   ELSE
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") "Sending message from rank ", my_rank, " to rank", 0
      CALL MPI_Ibsend(sbuffer, nints, MPI_INTEGER, 0, my_rank, MPI_COMM_WORLD, myrequest, ierr)
      CALL MPI_Wait(myrequest, mystatus, ierr)
   END IF

   CALL MPI_Buffer_detach(buffer_c_addr, bufsize, ierr)

   DEALLOCATE(sbuffer)
   DEALLOCATE(rbuffer)
   DEALLOCATE(buffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM ibsend_recv

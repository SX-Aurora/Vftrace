PROGRAM igather_inplace

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:,:), ALLOCATABLE :: rbuffer

   INTEGER, PARAMETER :: rootrank = 0
   INTEGER :: irank

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./igather_inplace <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   IF (my_rank == rootrank) THEN
      ALLOCATE(rbuffer(nints,comm_size))
      rbuffer(:,:) = my_rank
   ELSE
      ALLOCATE(sbuffer(nints))
      sbuffer(:) = my_rank
      ALLOCATE(rbuffer(0,0))
   END IF

   ! Message
   IF (my_rank == rootrank) THEN
      CALL MPI_Igather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &
                       rbuffer, nints, MPI_INTEGER, &
                       rootrank, MPI_COMM_WORLD, &
                       myrequest, ierr)
   ELSE
      CALL MPI_Igather(sbuffer, nints, MPI_INTEGER, &
                       rbuffer, nints, MPI_INTEGER, &
                       rootrank, MPI_COMM_WORLD, &
                       myrequest, ierr)
   END IF
   CALL MPI_Wait(myrequest, mystatus, ierr)

   IF (my_rank == rootrank) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") &
         "Gathering messages from all ranks on rank ", my_rank
   END IF

   ! validate data
   valid_data = .TRUE.
   IF (my_rank == rootrank) THEN
      DO irank = 0, comm_size - 1
         IF (ANY(rbuffer(:,irank+1) /= irank)) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
               "Rank ", my_rank, " received faulty data from rank ", irank
            valid_data = .FALSE.
         END IF
      END DO
   ELSE
      DEALLOCATE(sbuffer)
   END IF

   DEALLOCATE(rbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM igather_inplace

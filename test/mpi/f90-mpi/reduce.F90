PROGRAM reduce

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer

   INTEGER, PARAMETER :: rootrank = 0

   INTEGER :: refresult

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./reduce <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(sbuffer(nints))
   sbuffer(:) = my_rank
   IF (my_rank == rootrank) THEN
      ALLOCATE(rbuffer(nints))
      rbuffer(:) = -1
   ELSE
      ALLOCATE(rbuffer(0))
   END IF

   ! Message cycle
   CALL MPI_Reduce(sbuffer, rbuffer, nints, MPI_INTEGER, &
                   MPI_SUM, rootrank, MPI_COMM_WORLD, ierr)

   IF (my_rank == rootrank) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") &
         "Reducing messages from all ranks on rank ", my_rank
   END IF

   ! validate data
   valid_data = .TRUE.
   IF (my_rank == rootrank) THEN
      refresult = ((comm_size-1)*comm_size)/2
      IF (ANY(rbuffer /= refresult)) THEN
         WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
            "Rank ", my_rank, " received faulty data"
         valid_data = .FALSE.
      END IF
   END IF

   DEALLOCATE(rbuffer)
   DEALLOCATE(sbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM reduce

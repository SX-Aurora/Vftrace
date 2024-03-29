PROGRAM scatter

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:,:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer

   INTEGER, PARAMETER :: rootrank = 0

   INTEGER :: irank

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./scatter <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(rbuffer(nints))
   rbuffer(:) = -1
   IF (my_rank == rootrank) THEN
      ALLOCATE(sbuffer(nints,comm_size))
      DO irank = 0, comm_size - 1
         sbuffer(:,irank+1) = irank
      END DO
   ELSE
      ALLOCATE(sbuffer(0,0))
   END IF

   ! Messageing
   CALL MPI_Scatter(sbuffer, nints, MPI_INTEGER, &
                    rbuffer, nints, MPI_INTEGER, &
                    rootrank, MPI_COMM_WORLD, ierr)

   IF (my_rank == rootrank) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") &
         "Scattering messages from rank ", my_rank
   END IF

   ! validate data
   valid_data = .TRUE.
   IF (ANY(rbuffer(:) /= my_rank)) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") &
         "Rank ", my_rank, " received faulty data from rank ", rootrank
      valid_data = .FALSE.
   END IF

   DEALLOCATE(rbuffer)
   DEALLOCATE(sbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM scatter

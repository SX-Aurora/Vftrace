PROGRAM reduce_scatter_block_inplace

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer

   LOGICAL :: valid_data

   INTEGER :: irank, i

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./reduce_scatter_block_inplace <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(rbuffer(comm_size*nints))
   DO irank = 0, comm_size-1
      DO i = 0, nints-1
         rbuffer(irank*nints+i+1) = irank
      END DO
   END DO

   ! Message cycle
   CALL MPI_Reduce_scatter_block(MPI_IN_PLACE, rbuffer, nints, MPI_INTEGER, &
                                 MPI_SUM, MPI_COMM_WORLD, ierr)

   WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") &
      "Reducing and scattering messages from all ranks to all ranks on rank", my_rank

   ! validate data
   valid_data = .TRUE.
   IF (ANY(rbuffer(1:nints) /= comm_size*my_rank)) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
         "Rank ", my_rank, " received faulty data"
      valid_data = .FALSE.
   END IF

   DEALLOCATE(rbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM reduce_scatter_block_inplace

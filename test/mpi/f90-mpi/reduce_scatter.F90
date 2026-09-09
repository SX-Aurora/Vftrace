PROGRAM reduce_scatter

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER :: ntots = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: recvcounts

   LOGICAL :: valid_data

   INTEGER :: irank, i, idx

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./reduce_scatter <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ntots = comm_size*nints+((comm_size-1)*comm_size)/2
   ALLOCATE(recvcounts(comm_size))
   DO irank = 0, comm_size-1
      recvcounts(irank+1) = nints + irank
   END DO

   ALLOCATE(sbuffer(ntots))
   idx = 1
   DO irank = 0, comm_size-1
      DO i = 0, recvcounts(irank+1)-1
         sbuffer(idx) = irank
         idx = idx + 1
      END DO
   END DO
   ALLOCATE(rbuffer(recvcounts(my_rank+1)))
   rbuffer(:) = -1

   ! Message cycle
   CALL MPI_Reduce_scatter(sbuffer, rbuffer, recvcounts, MPI_INTEGER, &
                           MPI_SUM, MPI_COMM_WORLD, ierr)

   WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") &
      "Reducing and scattering messages from all ranks to all ranks on rank", my_rank

   ! validate data
   valid_data = .TRUE.
   IF (ANY(rbuffer /= comm_size*my_rank)) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
         "Rank ", my_rank, " received faulty data"
      valid_data = .FALSE.
   END IF

   DEALLOCATE(rbuffer)
   DEALLOCATE(sbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM reduce_scatter

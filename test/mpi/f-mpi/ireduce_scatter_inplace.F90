PROGRAM ireduce_scatter_inplace

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER :: ntots = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: recvcounts

   INTEGER, PARAMETER :: rootrank = 0

   INTEGER :: refresult

   LOGICAL :: valid_data

   INTEGER :: myrequest
   INTEGER :: mystatus(MPI_STATUS_SIZE)

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./ireduce_scatter_inplace <msgsize in integers>"
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

   ALLOCATE(rbuffer(ntots))
   idx = 1
   DO irank = 0, comm_size-1
      DO i = 0, recvcounts(irank+1)-1
         rbuffer(idx) = irank
         idx = idx + 1
      END DO
   END DO

   ! Message cycle
   CALL MPI_Ireduce_scatter(MPI_IN_PLACE, rbuffer, recvcounts, MPI_INTEGER, &
                           MPI_SUM, MPI_COMM_WORLD, myrequest, ierr)

   WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") &
      "Reducing and scattering messages from all ranks to all ranks on rank", my_rank

   CALL MPI_Wait(myrequest, mystatus, ierr)

   ! validate data
   valid_data = .TRUE.
   IF (ANY(rbuffer(1:recvcounts(my_rank+1)) /= comm_size*my_rank)) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
         "Rank ", my_rank, " received faulty data"
      valid_data = .FALSE.
   END IF

   DEALLOCATE(rbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM ireduce_scatter_inplace

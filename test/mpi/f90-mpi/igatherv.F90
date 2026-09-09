PROGRAM igatherv

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER :: ntot
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: recvcounts
   INTEGER, DIMENSION(:), ALLOCATABLE :: displs

   INTEGER, PARAMETER :: rootrank = 0
   INTEGER :: irank, i

   LOGICAL :: valid_data

   INTEGER :: myrequest
   INTEGER :: mystatus(MPI_STATUS_SIZE)

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./igatherv <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   nints = nints + my_rank
   ALLOCATE(sbuffer(nints))
   sbuffer(:) = my_rank
   IF (my_rank == rootrank) THEN
      ALLOCATE(recvcounts(comm_size))
      ALLOCATE(displs(comm_size))
      ntot = 0
      DO irank = 0, comm_size - 1
         recvcounts(irank+1) = nints+irank
         displs(irank+1) = ntot
         ntot = ntot + recvcounts(irank+1)
      END DO
      ALLOCATE(rbuffer(ntot))
      rbuffer(:) = -1
   ELSE
      ALLOCATE(recvcounts(0))
      ALLOCATE(displs(0))
      ALLOCATE(rbuffer(0))
   END IF

   ! Message
   CALL MPI_Igatherv(sbuffer, nints, MPI_INTEGER, &
                     rbuffer, recvcounts, displs, MPI_INTEGER, &
                     rootrank, MPI_COMM_WORLD, myrequest, ierr)
   CALL MPI_Wait(myrequest, mystatus, ierr)

   IF (my_rank == rootrank) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") &
         "Gathering messages from all ranks on rank ", my_rank
   END IF

   ! validate data
   valid_data = .TRUE.
   IF (my_rank == rootrank) THEN
      DO irank = 0, comm_size - 1
         DO i = 1, recvcounts(irank+1)
            IF (rbuffer(i+displs(irank+1)) /= irank) THEN
               WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
                  "Rank ", my_rank, " received faulty data from rank ", irank
               valid_data = .FALSE.
               EXIT
            END IF
         END DO
      END DO
   END IF

   DEALLOCATE(recvcounts)
   DEALLOCATE(displs)

   DEALLOCATE(rbuffer)
   DEALLOCATE(sbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM igatherv

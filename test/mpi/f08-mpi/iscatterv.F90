PROGRAM iscatterv

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER :: ntot = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: sendcounts
   INTEGER, DIMENSION(:), ALLOCATABLE :: displs

   INTEGER, PARAMETER :: rootrank = 0

   INTEGER :: irank, i

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./iscatterv <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   nints = nints + my_rank
   ALLOCATE(rbuffer(nints))
   rbuffer(:) = -1
   IF (my_rank == rootrank) THEN
      ALLOCATE(sendcounts(comm_size))
      ALLOCATE(displs(comm_size))
      ntot = 0
      DO irank = 0, comm_size - 1
         sendcounts(irank+1) = nints + irank
         displs(irank+1) = ntot
         ntot = ntot + sendcounts(irank+1)
      END DO
      ALLOCATE(sbuffer(ntot))
      DO irank = 0, comm_size - 1
         DO i = 1, sendcounts(irank+1)
            sbuffer(i+displs(irank+1)) = irank
         END DO
      END DO
   ELSE
      ALLOCATE(sendcounts(0))
      ALLOCATE(displs(0))
      ALLOCATE(sbuffer(0))
   END IF

   ! Messageing
   CALL MPI_Iscatterv(sbuffer, sendcounts, displs, MPI_INTEGER, &
                      rbuffer, nints, MPI_INTEGER, &
                      rootrank, MPI_COMM_WORLD, myrequest, ierr)
   CALL MPI_Wait(myrequest, mystatus, ierr)

   IF (my_rank == rootrank) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") &
         "Scattering messages to all ranks from rank ", my_rank
   END IF

   ! validate data
   valid_data = .TRUE.
   IF (ANY(rbuffer(:) /= my_rank)) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") &
         "Rank ", my_rank, " received faulty data from rank ", rootrank
      valid_data = .FALSE.
   END IF

   DEALLOCATE(sendcounts)
   DEALLOCATE(displs)

   DEALLOCATE(rbuffer)
   DEALLOCATE(sbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM iscatterv

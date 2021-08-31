PROGRAM alltoallv

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER :: nstot, nrtot
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: scounts
   INTEGER, DIMENSION(:), ALLOCATABLE :: sdispls
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: rcounts
   INTEGER, DIMENSION(:), ALLOCATABLE :: rdispls

   INTEGER :: irank, i

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./alltoallv <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   nints = nints + my_rank

   ! prepare special arrays for send
   ALLOCATE(scounts(comm_size))
   ALLOCATE(sdispls(comm_size))
   nstot = 0
   DO irank = 0, comm_size - 1
      scounts(irank+1) = nints
      sdispls(irank+1) = nstot
      nstot = nstot + scounts(irank+1)
   END DO
   ALLOCATE(sbuffer(nstot))
   sbuffer(:) = my_rank

   ! prepare special arrays for recv
   ALLOCATE(rcounts(comm_size))
   ALLOCATE(rdispls(comm_size))
   nrtot = 0
   DO irank = 0, comm_size - 1
      rcounts(irank+1) = nints - my_rank + irank
      rdispls(irank+1) = nrtot
      nrtot = nrtot + rcounts(irank+1)
   END DO
   ALLOCATE(rbuffer(nrtot))
   rbuffer(:) = -1

   ! Messaging
   CALL MPI_Alltoallv(sbuffer, scounts, sdispls, MPI_INTEGER, &
                      rbuffer, rcounts, rdispls, MPI_INTEGER, &
                      MPI_COMM_WORLD, ierr)

   WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") &
      "Communicating with all ranks"

   ! validate data
   valid_data = .TRUE.
   DO irank = 0, comm_size - 1
      DO i = 1, rcounts(irank+1)
         IF (rbuffer(i+rdispls(irank+1)) /= irank) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") &
               "Rank ", my_rank, " received faulty data from rank ", irank
            valid_data = .FALSE.
            EXIT
         END IF
      END DO
   END DO
   DEALLOCATE(rbuffer)

   DEALLOCATE(rcounts)
   DEALLOCATE(rdispls)

   DEALLOCATE(sbuffer)

   DEALLOCATE(scounts)
   DEALLOCATE(sdispls)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM alltoallv

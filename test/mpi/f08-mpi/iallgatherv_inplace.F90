PROGRAM iallgatherv_inplace

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi_f08

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER :: ntot
   INTEGER, DIMENSION(:), ALLOCATABLE :: rbuffer
   INTEGER, DIMENSION(:), ALLOCATABLE :: recvcounts
   INTEGER, DIMENSION(:), ALLOCATABLE :: displs

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./iallgatherv_inplace <msgsize in integers>"
      STOP 1
   END IF

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   nints = nints + my_rank
   ALLOCATE(recvcounts(comm_size))
   ALLOCATE(displs(comm_size))
   ntot = 0
   DO irank = 0, comm_size - 1
      recvcounts(irank+1) = nints - my_rank + irank
      displs(irank+1) = ntot
      ntot = ntot + recvcounts(irank+1)
   END DO
   ALLOCATE(rbuffer(ntot))
   rbuffer(:) = my_rank

   ! Message
   CALL MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &
                        rbuffer, recvcounts, displs, MPI_INTEGER, &
                        MPI_COMM_WORLD, myrequest, ierr)
   CALL MPI_Wait(myrequest, mystatus, ierr)

   WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") &
      "Gathering messages from all ranks on rank ", my_rank

   ! validate data
   valid_data = .TRUE.
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
   DEALLOCATE(rbuffer)

   DEALLOCATE(recvcounts)
   DEALLOCATE(displs)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM iallgatherv_inplace

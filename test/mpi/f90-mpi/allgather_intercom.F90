PROGRAM allgather_intercom

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: sbuffer
   INTEGER, DIMENSION(:,:), ALLOCATABLE :: rbuffer

   INTEGER :: color
   INTEGER :: sub_comm
   INTEGER :: sub_comm_size
   INTEGER :: my_sub_rank

   INTEGER :: int_comm
   INTEGER :: local_leader, remote_leader
   INTEGER :: sub_comm_remote_size

   INTEGER :: irank, jrank

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./allgather_intercom <msgsize in integers>"
      STOP 1
   END IF

   ! create intercommunicator
   color = (2*my_rank) / comm_size
   CALL MPI_Comm_split(MPI_COMM_WORLD, &
                       color, my_rank, sub_comm, ierr)
   ! get local comm size and rank
   CALL MPI_Comm_size(sub_comm, sub_comm_size, ierr)
   CALL MPI_Comm_rank(sub_comm, my_sub_rank, ierr)

   local_leader = 0
   remote_leader = (1-color)*(comm_size+1)/2
   CALL MPI_Intercomm_create(sub_comm, &
                             local_leader, &
                             MPI_COMM_WORLD, &
                             remote_leader, 1, &
                             int_comm, ierr)
   CALL MPI_Comm_remote_size(int_comm, sub_comm_remote_size, ierr)

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   ALLOCATE(sbuffer(nints))
   sbuffer(:) = my_rank
   ALLOCATE(rbuffer(nints,sub_comm_remote_size))
   rbuffer(:,:) = -1

   CALL MPI_AllGather(sbuffer, nints, MPI_INTEGER, &
                      rbuffer, nints, MPI_INTEGER, &
                      int_comm, ierr)
   WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") &
      "Gathering messages on rank ", my_rank, "from remote group"

   ! validate data
   valid_data = .TRUE.
   DO irank = 0, sub_comm_remote_size - 1
      jrank = (1-color)*sub_comm_size + irank
      IF (ANY(rbuffer(:,irank+1) /= jrank)) THEN
         WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
            "Rank ", my_rank, " received faulty data from rank", jrank
         valid_data = .FALSE.
      END IF
   END DO

   DEALLOCATE(rbuffer)

   CALL MPI_Comm_free(int_comm, ierr)
   CALL MPI_Comm_free(sub_comm, ierr)

   DEALLOCATE(sbuffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM allgather_intercom

PROGRAM ibcast_intercom

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER :: nints = 0
   INTEGER, DIMENSION(:), ALLOCATABLE :: buffer

   INTEGER :: color
   INTEGER :: sub_comm
   INTEGER :: sub_comm_size
   INTEGER :: my_sub_rank

   INTEGER :: int_comm
   INTEGER :: local_leader, remote_leader
   INTEGER :: sub_comm_remote_size

   INTEGER :: root

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./ibcast_intercom <msgsize in integers>"
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
   ALLOCATE(buffer(nints))
   buffer(:) = my_rank

   ! prepare the intercomm root assignment
   IF (color == 0) THEN
      ! sub communicator of receiving group
      IF (my_sub_rank == 0) THEN
         ! Receiving rank in receiving subgroup
         root = MPI_ROOT
      ELSE
         ! Ideling processes
         root = MPI_PROC_NULL
      END IF
   ELSE
      ! Sub communicator or sending group
      root = 0 ! receiving rank in remote group
   END IF

   CALL MPI_Ibcast(buffer, nints, MPI_INTEGER, root, int_comm, myrequest, ierr)
   CALL MPI_Wait(myrequest, mystatus, ierr)

   IF (my_rank == 0) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4)") &
         "Broadcasted message from global rank ", my_rank
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4,A)") &
         "(Group=",color,", local rank=", my_sub_rank,")"
   END IF

   ! validate data
   valid_data = .TRUE.
   SELECT CASE (root)
      CASE (MPI_ROOT)
         IF (ANY(buffer /= my_rank)) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
               "Root process (Rank ", my_rank, ") has no longer valid data"
            valid_data = .FALSE.
         END IF
      CASE (MPI_PROC_NULL)
         IF (ANY(buffer /= my_rank)) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A)") &
               "Uninvolved process (Rank ", my_rank, ") has no longer valid data"
            valid_data = .FALSE.
         END IF
      CASE DEFAULT
         IF (ANY(buffer /= root)) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") &
               "Rank ", my_rank, " received faulty data from rank ", root
            valid_data = .FALSE.
         END IF
   END SELECT

   CALL MPI_Comm_free(int_comm, ierr)
   CALL MPI_Comm_free(sub_comm, ierr)

   DEALLOCATE(buffer)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM ibcast_intercom

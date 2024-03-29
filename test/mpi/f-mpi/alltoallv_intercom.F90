PROGRAM alltoallv_intercom

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

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

   INTEGER :: color
   INTEGER :: sub_comm
   INTEGER :: sub_comm_size
   INTEGER :: my_sub_rank

   INTEGER :: int_comm
   INTEGER :: local_leader, remote_leader
   INTEGER :: sub_comm_remote_size
   INTEGER :: minpeerrank

   INTEGER :: irank, jrank, i

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
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./alltoallv_intercom <msgsize in integers>"
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

   minpeerrank = (1-color)*((comm_size+1)/2)

   ! Allocating send/recv buffer
   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nints
   nints = nints + my_rank

   ! prepareing the send intercom special arrays
   ALLOCATE(scounts(sub_comm_remote_size))
   ALLOCATE(sdispls(sub_comm_remote_size))
   nstot = 0
   DO irank = 0, sub_comm_remote_size - 1
      jrank = minpeerrank + irank
      scounts(irank+1) = nints
      sdispls(irank+1) = nstot
      nstot = nstot + scounts(irank+1)
   END DO
   ALLOCATE(sbuffer(nstot))
   sbuffer(:) = my_rank

   ! preparing the recv intercom special arrays
   ALLOCATE(rcounts(sub_comm_remote_size))
   ALLOCATE(rdispls(sub_comm_remote_size))
   nrtot = 0
   DO irank = 0, sub_comm_remote_size - 1
      jrank = minpeerrank + irank
      rcounts(irank+1) = nints - my_rank + jrank
      rdispls(irank+1) = nrtot
      nrtot = nrtot + rcounts(irank+1)
   END DO
   ALLOCATE(rbuffer(nrtot))
   rbuffer(:) = -1

   CALL MPI_Alltoallv(sbuffer, scounts, sdispls, MPI_INTEGER, &
                      rbuffer, rcounts, rdispls, MPI_INTEGER, &
                      int_comm, ierr)
   WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") &
      "Communicating with all ranks"

   ! validate data
   valid_data = .TRUE.
   DO irank = 0, sub_comm_remote_size - 1
      jrank = minpeerrank + irank
      DO i = 1, rcounts(irank+1)
         IF (rbuffer(i+rdispls(irank+1)) /= jrank) THEN
            WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I4,A,I4)") &
               "Rank ", my_rank, " received faulty data from rank", jrank
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

   CALL MPI_Comm_free(int_comm, ierr)
   CALL MPI_Comm_free(sub_comm, ierr)

   CALL MPI_Finalize(ierr)

   IF (.NOT.valid_data) STOP 1
END PROGRAM alltoallv_intercom

PROGRAM ftypes

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER          :: send_integer,          recv_integer
   REAL             :: send_real,             recv_real
   DOUBLE PRECISION :: send_double_precision, recv_double_precision
   COMPLEX          :: send_complex,          recv_complex
   LOGICAL          :: send_logical,          recv_logical
   CHARACTER(LEN=1) :: send_character,        recv_character

   INTEGER :: recvstatus(MPI_STATUS_SIZE)

   INTEGER :: ierr

   CALL MPI_Init(ierr)

   ! Get the number of processes
   CALL MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr);
   ! Get rank of processes
   CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr);

   ! Write information
   IF (comm_size < 2) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I6,A)") "At least two ranks are required"
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A,I6,A)") "RUN again with '-np 2'"
      CALL FLUSH(OUTPUT_UNIT)
      CALL MPI_Finalize(ierr)
      STOP
   END IF

   ! Messaging
   IF (my_rank == 0) THEN
      ! sending rank
      CALL MPI_Send(send_integer, 1, MPI_INTEGER, 1, 0, MPI_COMM_WORLD, ierr)
      CALL MPI_Send(send_real, 1, MPI_REAL, 1, 0, MPI_COMM_WORLD, ierr)
      CALL MPI_Send(send_double_precision, 1, MPI_DOUBLE_PRECISION, 1, 0, MPI_COMM_WORLD, ierr)
      CALL MPI_Send(send_complex, 1, MPI_COMPLEX, 1, 0, MPI_COMM_WORLD, ierr)
      CALL MPI_Send(send_logical, 1, MPI_LOGICAL, 1, 0, MPI_COMM_WORLD, ierr)
      CALL MPI_Send(send_character, 1, MPI_CHARACTER, 1, 0, MPI_COMM_WORLD, ierr)
   ELSE IF (my_rank == 1) THEN
      CALL MPI_Recv(recv_integer, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, recvstatus, ierr)
      CALL MPI_Recv(recv_real, 1, MPI_REAL, 0, 0, MPI_COMM_WORLD, recvstatus, ierr)
      CALL MPI_Recv(recv_double_precision, 1, MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD, recvstatus, ierr)
      CALL MPI_Recv(recv_complex, 1, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, recvstatus, ierr)
      CALL MPI_Recv(recv_logical, 1, MPI_LOGICAL, 0, 0, MPI_COMM_WORLD, recvstatus, ierr)
      CALL MPI_Recv(recv_character, 1, MPI_CHARACTER, 0, 0, MPI_COMM_WORLD, recvstatus, ierr)
   END IF

   CALL MPI_Finalize(ierr)

END PROGRAM ftypes

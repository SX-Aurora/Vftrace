PROGRAM derived_ftypes

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE mpi

   IMPLICIT NONE

   TYPE testtype
      INTEGER :: aninteger1, aninteger2
      CHARACTER(LEN=16) :: string
      DOUBLE PRECISION :: adouble1, adouble2, adouble3
      INTEGER, DIMENSION(32) :: integerarr
   END TYPE testtype

   INTEGER :: comm_size
   INTEGER :: my_rank

   INTEGER, PARAMETER :: nelem = 1024
   TYPE(testtype), DIMENSION(nelem) :: srbuffer
   ! MPI_datatypes
   INTEGER :: mpi_chararr_t
   INTEGER :: mpi_intarr_t
   INTEGER :: mpi_testtype_t
   INTEGER :: blocklength(4)
   INTEGER(KIND=MPI_ADDRESS_KIND) :: displacements(4)
   INTEGER :: types(4)

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

   ! construct the MPI_Datatypes for a nested construct
   ! datatype for the character of length 16
   CALL MPI_Type_contiguous(16, MPI_CHARACTER, mpi_chararr_t, ierr)
   CALL MPI_Type_commit(mpi_chararr_t, ierr)
   ! datatype for the int array of size 32
   CALL MPI_Type_contiguous(32, MPI_INTEGER, mpi_intarr_t, ierr)
   CALL MPI_Type_commit(mpi_chararr_t, ierr)
   ! create the derived fortran type for mpi
   blocklength = [2,1,3,1]
   displacements = [0,8,24,48]
   types = [MPI_INTEGER, mpi_chararr_t, MPI_DOUBLE_PRECISION, mpi_intarr_t]
   CALL MPI_Type_create_struct(4, blocklength, displacements, types, mpi_testtype_t, ierr)
   CALL MPI_Type_commit(mpi_testtype_t, ierr)

   ! Messaging
   IF (my_rank == 0) THEN
      ! sending rank
      CALL MPI_Send(srbuffer, nelem, mpi_testtype_t, 1, 0, MPI_COMM_WORLD, ierr)
      CALL MPI_Recv(srbuffer, nelem, mpi_testtype_t, 1, 0, MPI_COMM_WORLD, recvstatus, ierr)
   ELSE IF (my_rank == 1) THEN
      CALL MPI_Recv(srbuffer, nelem, mpi_testtype_t, 0, 0, MPI_COMM_WORLD, recvstatus, ierr)
      CALL MPI_Send(srbuffer, nelem, mpi_testtype_t, 0, 0, MPI_COMM_WORLD, ierr)
   END IF

   CALL MPI_Type_free(mpi_testtype_t, ierr)
   CALL MPI_Type_free(mpi_intarr_t, ierr)
   CALL MPI_Type_free(mpi_chararr_t, ierr)

   CALL MPI_Finalize(ierr)

END PROGRAM derived_ftypes

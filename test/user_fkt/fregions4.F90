PROGRAM fregions4

   USE, INTRINSIC :: ISO_FORTRAN_ENV
#ifdef _MPI
   USE mpi
#endif

   USE vftrace

   IMPLICIT NONE

   CHARACTER(LEN=64) :: cmdargstr = ""

   CHARACTER(LEN=32) :: reg_name
   CHARACTER(LEN=8) :: reg_num

   INTEGER :: nreg
   INTEGER :: ireg

#ifdef _MPI
   INTEGER :: ierr

   CALL MPI_Init(ierr)
#endif

   ! require cmd-line argument
   IF (COMMAND_ARGUMENT_COUNT() < 1) THEN
      WRITE(UNIT=OUTPUT_UNIT, FMT="(A)") "./fregions4 <nregions>"
      STOP 1
   END IF

   CALL GET_COMMAND_ARGUMENT(1,cmdargstr)
   READ(UNIT=cmdargstr, FMT=*) nreg

   DO ireg = 1, nreg
      WRITE(UNIT=reg_num, FMT='(I8)') ireg
      reg_name = "user-region-"//TRIM(ADJUSTL(reg_num))
      WRITE(*,'(A)') reg_name
      CALL vftrace_region_begin(reg_name)
   END DO
   DO ireg = nreg, 1, -1
      WRITE(UNIT=reg_num, FMT='(I8)') ireg
      reg_name = "user-region-"//TRIM(ADJUSTL(reg_num))
      WRITE(*,'(A)') reg_name
      CALL vftrace_region_end(reg_name)
   END DO

#ifdef _MPI
   CALL MPI_Finalize(ierr)
#endif
END PROGRAM fregions4

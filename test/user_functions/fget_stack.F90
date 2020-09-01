PROGRAM fget_stack
#ifdef _MPI
   USE mpi
#endif

   IMPLICIT NONE

   INTEGER :: sum

#ifdef _MPI
   INTEGER :: ierr

   CALL MPI_Init(ierr)
#endif

   CALL fkt1()

#ifdef _MPI
   CALL MPI_Finalize(ierr)
#endif
END PROGRAM fget_stack

SUBROUTINE fkt1()
   CALL fkt2()
   RETURN
END SUBROUTINE fkt1
SUBROUTINE fkt2()
   CALL fkt3()
   RETURN
END SUBROUTINE fkt2
SUBROUTINE fkt3()
   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE vftrace
   WRITE(UNIT=OUTPUT_UNIT, FMT='(A)') vftrace_get_stack()
   RETURN
END SUBROUTINE fkt3

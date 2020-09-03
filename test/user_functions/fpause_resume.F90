PROGRAM fpause_resume
#ifdef _MPI
   USE mpi
#endif

   USE vftrace

   IMPLICIT NONE

   INTEGER :: fkt1, fkt2, fkt3
   INTEGER :: sum

#ifdef _MPI
   INTEGER :: ierr

   CALL MPI_Init(ierr)
#endif

   sum = 0
   ! This code is profiled
   sum = sum + fkt1()

   CALL vftrace_pause()
   ! This code is not profiled
   sum = sum + fkt2()

   CALL vftrace_resume()
   ! This code is profiled again
   sum = sum + fkt3()

   IF (sum /= 6) STOP 1

#ifdef _MPI
   CALL MPI_Finalize(ierr)
#endif
END PROGRAM fpause_resume

INTEGER FUNCTION fkt1()
   fkt1 = 1
   RETURN
END FUNCTION fkt1
INTEGER FUNCTION fkt2()
   fkt2 = 2
   RETURN
END FUNCTION fkt2
INTEGER FUNCTION fkt3()
   fkt3 = 3
   RETURN
END FUNCTION fkt3

program fshow_stack1
#ifdef _MPI
   use mpi
#endif

   implicit none

#ifdef _MPI
   integer :: ierr
   call MPI_Init (ierr)
#endif

   call func1()
   call func2() 

#ifdef _MPI
   call MPI_Finalize (ierr)
#endif

end program fshow_stack1

subroutine func1()
  call func2()
end subroutine

subroutine func2()
  use vftrace
  call vftrace_show_callstack()
end subroutine

#include <stdio.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <vftrace.h>

int fkt3() {
   printf("%s\n", vftrace_get_stack());
   return 1;
}
int fkt2() {
   return fkt3();
}
int fkt1() {
   return fkt2();
}

int main() {

#ifdef _MPI
   MPI_Init(NULL, NULL);
#endif

   fkt1();

#ifdef _MPI
   MPI_Finalize();
#endif
   return 0;
}


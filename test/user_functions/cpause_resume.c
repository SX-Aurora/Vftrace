#ifdef _MPI
#include <mpi.h>
#endif

#include <vftrace.h>

int fkt1() {
   return 1;
}
int fkt2() {
   return 2;
}
int fkt3() {
   return 3;
}

int main() {

#ifdef _MPI
   MPI_Init(NULL, NULL);
#endif

   int sum = 0;

   // This code is profiled
   sum += fkt1();

   vftrace_pause();
   // This code is not profiled
   sum += fkt2();

   vftrace_resume();
   // This code is profiled again
   sum += fkt3();

#ifdef _MPI
   MPI_Finalize();
#endif

   return sum == 6 ? 0 : 1;
}


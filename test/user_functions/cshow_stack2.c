#include <stdlib.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <vftrace.h>

int func1(int x);
int func2(int x);

int func1(int x) {
  vftrace_show_callstack();
  return x;
}

int func2(int x) {
  vftrace_show_callstack();
  return x;
}

int main() {

#ifdef _MPI
   MPI_Init (NULL, NULL);
#endif

   int x1 = func1(1);
   int x2 = func2(1);

#ifdef _MPI
   MPI_Finalize();
#endif
}

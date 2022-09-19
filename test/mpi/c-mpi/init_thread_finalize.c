#include <mpi.h>
#include <stdlib.h>

int main() {
   int provided;
   MPI_Init_thread(NULL,NULL,MPI_THREAD_SINGLE,&provided);
   MPI_Finalize();
   return 0;
}

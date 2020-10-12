#include <mpi.h>
#include <stdlib.h>

int main() {
   MPI_Init(NULL,NULL);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
   return 0;
}

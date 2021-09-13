#include <mpi.h>
#include <stdlib.h>

int main() {
   MPI_Init(NULL,NULL);
   MPI_Request request;
   MPI_Status status;
   MPI_Ibarrier(MPI_COMM_WORLD, &request);
   MPI_Wait(&request, &status);
   MPI_Finalize();
   return 0;
}

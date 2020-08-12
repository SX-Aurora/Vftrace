#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv) {

   MPI_Init(&argc, &argv);

   // Get number or processes
   int comm_size;
   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
   // Get rank of process
   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   // require at least two processes
   if (comm_size < 2) {
      printf("At least two ranks are required\n");
      printf("Run again with '-np 2'\n");
      MPI_Finalize();
      return 1;
   }

   // require cmd-line argument
   if (argc < 2) {
      printf("./wait <msgsize in Byte>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nbyte = atoi(argv[1]);
   void* srbuffer = malloc(nbyte);

   // Messaging
   if (my_rank == 0) {
      MPI_Request request;
      // send to other rank
      MPI_Isend(srbuffer, nbyte, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &request);
      MPI_Status mystat;
      MPI_Wait(&request, &mystat);
      printf("Sending request completed\n");
   } else if (my_rank == 1){
      sleep(my_rank);
      MPI_Status mystat;
      MPI_Recv(srbuffer, nbyte, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &mystat);
   }

   free(srbuffer);
   srbuffer=NULL;

   MPI_Finalize();

   return 0;
}

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
      printf("./testall <msgsize in Byte>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nbyte = atoi(argv[1]);
   void* srbuffer = malloc(nbyte);

   // Messaging
   if (my_rank == 0) {
      MPI_Request *requests = (MPI_Request*) malloc((comm_size-1)*sizeof(MPI_Request));
      MPI_Status *statuses = (MPI_Status*) malloc((comm_size-1)*sizeof(MPI_Status));
      // send to every other rank
      for (int recvrank=1; recvrank<comm_size; recvrank++) {
         MPI_Isend(srbuffer, nbyte, MPI_BYTE, recvrank, 0, MPI_COMM_WORLD, requests+recvrank-1);
      }
      MPI_Waitall(comm_size-1, requests, statuses);
      printf("All requests are completed\n");
      free(requests);
      requests = NULL;
      free(statuses);
      statuses = NULL;
   } else {
      MPI_Status mystat;
      sleep(2*(my_rank%2));
      MPI_Recv(srbuffer, nbyte, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &mystat);
   }

   free(srbuffer);
   srbuffer=NULL;

   MPI_Finalize();

   return 0;
}

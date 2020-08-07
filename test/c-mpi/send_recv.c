#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv) {
   MPI_Init(NULL, NULL);

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
      printf("./send_recv <msgsize in kiB>\n");
      return 1;
   }

   // allocating send/recv buffer
   const int nbyte = atoi(argv[1]);
   void* srbuffer = malloc(nbyte);

   // Messaging cycle
   for (int sendrank=0; sendrank<comm_size; sendrank++) {
      if (my_rank == sendrank) {
         // send to every other rank
         for (int recvrank=0; recvrank<comm_size; recvrank++) {
            if (my_rank != recvrank) {
               printf("Sending messages from rank %d\n", my_rank);
               MPI_Send(srbuffer, nbyte, MPI_BYTE, recvrank, 0, MPI_COMM_WORLD);
            }
         }
      } else {
         MPI_Status mystat;
         printf("Sending messages from rank %d\n", sendrank);
         MPI_Recv(srbuffer, nbyte, MPI_BYTE, sendrank, 0, MPI_COMM_WORLD, &mystat);
      }
   }

   free(srbuffer);
   srbuffer=NULL;

   MPI_Finalize();

   return 0;
}

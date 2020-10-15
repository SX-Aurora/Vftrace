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
      printf("./test <msgsize in Byte>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nbyte = atoi(argv[1]);
   void* srbuffer = malloc(nbyte);

   // Messaging
   if (my_rank == 0) {
      MPI_Request request;
      // send to every rank
      MPI_Isend(srbuffer, nbyte, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &request);
      int flag = 0;
      while (!flag) {
         MPI_Status mystat;
         MPI_Test(&request, &flag, &mystat);
         if (flag) {
            printf("Sending request is completed\n");
         } else {
            printf("Sending request is not completed\n");
            sleep(1);
         }
      }
   } else if (my_rank == 1) {
      MPI_Status mystat;
      sleep(2);
      MPI_Recv(srbuffer, nbyte, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &mystat);
   }
   MPI_Barrier(MPI_COMM_WORLD);

   free(srbuffer);
   srbuffer=NULL;

   MPI_Finalize();

   return 0;
}

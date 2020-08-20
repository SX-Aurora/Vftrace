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
      printf("./testany <msgsize in Byte>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nbyte = atoi(argv[1]);
   void* srbuffer = malloc(nbyte);

   // Messaging
   MPI_Status mystat;
   MPI_Barrier(MPI_COMM_WORLD);
   if (my_rank == 0) {
      MPI_Request *requests = (MPI_Request*) malloc((comm_size-1)*sizeof(MPI_Request));
      // send to every other rank
      for (int recvrank=1; recvrank<comm_size; recvrank++) {
         MPI_Isend(srbuffer, nbyte, MPI_BYTE, recvrank, 0, MPI_COMM_WORLD, requests+recvrank-1);
      }
      int idx;
      int flag=0;
      while (flag == 0) {
         MPI_Testany(comm_size-1, requests, &idx, &flag, &mystat);
         if (flag) {
            printf("Sending request to rank %d is completed\n", idx+1);
         } else {
            printf("No sending request is completed\n");
            sleep(1);
         }
      }
      for (int ireq=0; ireq<comm_size-1; ireq++) {
         if (ireq != idx) {
            MPI_Wait(requests+ireq, &mystat);
         }
      }
      free(requests);
      requests = NULL;
   } else {
      sleep(2*(my_rank%2));
      MPI_Recv(srbuffer, nbyte, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &mystat);
   }

   free(srbuffer);
   srbuffer=NULL;

   MPI_Finalize();

   return 0;
}

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
      printf("./isend_recv <msgsize in kiB>\n");
      return 1;
   }
   // allocating send/recv buffer
   const int nbyte = atoi(argv[1]);
   void* sbuffer = malloc(nbyte);
   void* rbuffer = malloc(nbyte);
   MPI_Request *myrequest = (MPI_Request*) malloc((comm_size-1)*sizeof(MPI_Request));

   // Messaging cycle
   for (int sendrank=0; sendrank<comm_size; sendrank++) {
      MPI_Status mystat;
      int reqidx = 0;
      if (my_rank == sendrank) {
         // send to every other rank
         for (int recvrank=0; recvrank<comm_size; recvrank++) {
            if (my_rank != recvrank) {
               printf("Sending messages from rank %d\n", my_rank);
               MPI_Isend(sbuffer, nbyte, MPI_BYTE, recvrank, 0, MPI_COMM_WORLD,
                         myrequest+reqidx);
               reqidx++;
            }
         }
         // wait for completion of non-blocking sends
         for (int ireq=0; ireq<comm_size-1; ireq++) {
            MPI_Wait(myrequest+ireq, &mystat);
         }
      } else {
         MPI_Recv(rbuffer, nbyte, MPI_BYTE, sendrank, 0, MPI_COMM_WORLD, &mystat);
      }
   }
   MPI_Barrier(MPI_COMM_WORLD);

   free(sbuffer);
   sbuffer=NULL;

   free(rbuffer);
   rbuffer=NULL;

   MPI_Finalize();

   return 0;
}

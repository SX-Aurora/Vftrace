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
      printf("./testsome <msgsize in Byte>\n");
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
      sleep(1);
      int outcount;
      int *idxs = (int*) malloc((comm_size-1)*sizeof(int));
      MPI_Waitsome(comm_size-1, requests, &outcount, idxs, statuses);
      for (int iidx=0; iidx<outcount; iidx++) {
         printf("Sending request to rank %d is completed\n", idxs[iidx]+1);
      }
      for (int ireq=0; ireq<comm_size-1; ireq++) {
         if (requests[ireq] != MPI_REQUEST_NULL) {
            MPI_Wait(requests+ireq, statuses);
         }
      }
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

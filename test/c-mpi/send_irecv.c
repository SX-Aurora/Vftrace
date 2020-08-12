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
      printf("./send_recv <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int* sbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {sbuffer[i]=my_rank;}
   int* rbuffer = (int*) malloc(comm_size*nints*sizeof(int));
   for (int i=0; i<comm_size*nints; i++) {rbuffer[i]=-1;}
   MPI_Request *requests = (MPI_Request*) malloc(comm_size*sizeof(MPI_Request));

   // prepare non-blocking receive
   for (int sendrank=0; sendrank<comm_size; sendrank++) {
      printf("Preparing message receiving from rank %d\n", sendrank);
      MPI_Irecv(rbuffer+sendrank*nints, nints, MPI_INT, sendrank,
                sendrank, MPI_COMM_WORLD, requests+sendrank);
   }

   // Sending
   for (int recvrank=0; recvrank<comm_size; recvrank++) {
      printf("Sending messages from rank %d\n", my_rank);
      MPI_Send(sbuffer, nints, MPI_INT, recvrank, my_rank, MPI_COMM_WORLD);
   }

   // wait for completion of Irecv
   for (int sendrank=0; sendrank<comm_size; sendrank++) {
      MPI_Status mystat;
      MPI_Wait(requests+sendrank, &mystat);
      printf("Received message from rank %d\n", sendrank);
   }

   // verify communication data
   bool valid_data = true;
   for (int sendrank=0; sendrank<comm_size; sendrank++) {
      for (int i=0; i<nints; i++) {
         if (rbuffer[i+sendrank*nints] != sendrank) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, sendrank);
            valid_data = false;
            break;
         }
      }
   }

   free(requests);
   requests = NULL;

   free(sbuffer);
   sbuffer=NULL;

   free(rbuffer);
   rbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

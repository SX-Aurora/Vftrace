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
   if (argc < 3) {
      printf("./send_init <msgsize in ints> <nrepetitions>\n");
      return 1;
   }

   int nruns = atoi(argv[2]);

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int* sbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {sbuffer[i]=my_rank;}
   int* rbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {rbuffer[i]=-1;}
   MPI_Request *myrequest = (MPI_Request*) malloc((comm_size-1)*sizeof(MPI_Request));

   // Messaging cycle
   bool valid_data = true;
   MPI_Status mystat;
   int reqidx = 0;
   if (my_rank == 0) {
      // prepare send to every other rank
      printf("Initialize sending messages from rank %d\n", my_rank);
      for (int recvrank=1; recvrank<comm_size; recvrank++) {
         MPI_Send_init(sbuffer, nints, MPI_INT, recvrank, 0, MPI_COMM_WORLD,
                       myrequest+reqidx);
         reqidx++;
      }

      for (int irun=0; irun<nruns; irun++) {
         // send to every other rank
         printf("Sending messages from rank %d\n", 0);
         for (int ireq=0; ireq<comm_size-1; ireq++) {
            MPI_Start(myrequest+ireq);
         }
         // wait for completion of non-blocking sends
         for (int ireq=0; ireq<comm_size-1; ireq++) {
            MPI_Wait(myrequest+ireq, &mystat);
         }
      }
   } else {
      for (int irun=0; irun<nruns; irun++) {
         int* rbuffer = (int*) malloc(nints*sizeof(int));
         printf("Receiving messages from rank %d\n", my_rank);
         MPI_Recv(rbuffer, nints, MPI_INT, 0, 0, MPI_COMM_WORLD, &mystat);
         // validate data
         for (int i=0; i<nints; i++) {
            if (rbuffer[i] != 0) {
               printf("Rank %d received faulty data from rank %d\n", my_rank, 0);
               valid_data = false;
               break;
            }
         }
      }
   }
   
   free(sbuffer);
   sbuffer=NULL;

   free(rbuffer);
   rbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

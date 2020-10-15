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
      printf("./start <msgsize in ints>\n");
      return 1;
   }

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
   if (my_rank == 0) {
      // prepare send to every other rank
      printf("Initialize sending messages from rank %d\n", my_rank);
      for (int recvrank=1; recvrank<comm_size; recvrank++) {
         MPI_Send_init(sbuffer, nints, MPI_INT, recvrank, 0, MPI_COMM_WORLD,
                       myrequest+recvrank-1);
      }

      // send to every other rank
      printf("Sending messages from rank %d\n", 0);
      for (int ireq=0; ireq<comm_size-1; ireq++) {
         MPI_Start(myrequest+ireq);
      }
      // mark persistent requests for deallocation
      // this is done here intentionally
      // to test the request free functionality
      //
      // half of the requests are freed the other half is waited for.
      for (int ireq=0; ireq<comm_size-1; ireq+=2) {
         MPI_Request_free(myrequest+ireq);
      }
      // wait for completion of non-blocking sends
      for (int ireq=1; ireq<comm_size-1; ireq+=2) {
         MPI_Wait(myrequest+ireq, &mystat);
      }
   } else {
      printf("Receiving messages from rank %d\n", 0);
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
   MPI_Barrier(MPI_COMM_WORLD);
   
   free(sbuffer);
   sbuffer=NULL;

   free(rbuffer);
   rbuffer=NULL;

   free(myrequest);
   myrequest=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

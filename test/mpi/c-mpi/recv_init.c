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
      printf("./recv_init <msgsize in ints> <nrepetitions>\n");
      return 1;
   }

   int nruns = atoi(argv[2]);

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int* sbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {sbuffer[i]=my_rank;}
   int* rbuffer = (int*) malloc((comm_size-1)*nints*sizeof(int));
   for (int i=0; i<nints*(comm_size-1); i++) {rbuffer[i]=-1;}
   MPI_Request *myrequest = (MPI_Request*) malloc((comm_size-1)*sizeof(MPI_Request));

   // Messaging cycle
   bool valid_data = true;
   if (my_rank == 0) {
      // prepare recv from every other rank
      printf("Initialize receiving messages on rank %d\n", my_rank);
      for (int sendrank=1; sendrank<comm_size; sendrank++) {
         MPI_Recv_init(rbuffer+nints*(sendrank-1), nints, MPI_INT, sendrank, sendrank, MPI_COMM_WORLD,
                       myrequest+sendrank-1);
      }

      for (int irun=0; irun<nruns; irun++) {
         for (int i=0; i<nints*(comm_size-1); i++) {
            rbuffer[i]=-1;
         }
         // recv from every other rank
         printf("Receiving messages on rank %d\n", 0);
         for (int ireq=0; ireq<comm_size-1; ireq++) {
            MPI_Start(myrequest+ireq);
         }
         // wait for completion of non-blocking receives
         for (int ireq=0; ireq<comm_size-1; ireq++) {
            MPI_Status mystat;
            MPI_Wait(myrequest+ireq, &mystat);
         }
         // validate data
         for (int irank=1; irank<comm_size; irank++) {
            for (int i=0; i<nints; i++) {
               if (rbuffer[(irank-1)*nints+i] != irank) {
                  printf("Rank %d received faulty data from rank %d\n", my_rank, irank);
                  valid_data = false;
                  break;
               }
            }
         }
      }
   } else {
      for (int irun=0; irun<nruns; irun++) {
         printf("Sending messages from rank %d\n", my_rank);
         MPI_Send(sbuffer, nints, MPI_INT, 0, my_rank, MPI_COMM_WORLD);
      }
   }
   
   free(sbuffer);
   sbuffer=NULL;

   free(rbuffer);
   rbuffer=NULL;

   free(myrequest);
   myrequest=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

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
      printf("./probe <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int* sbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {sbuffer[i]=my_rank;}
   int* rbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {rbuffer[i]=-1;}

   // Messaging cycle
   bool valid_data = true;
   if (my_rank == 0) {
      printf("Sending message from rank %d to rank %d\n", my_rank, 1);
      MPI_Send(sbuffer, nints, MPI_INT, 1, 0, MPI_COMM_WORLD);
   } else if (my_rank == 1) {
      MPI_Status mystat;
      printf("Receiving message on rank %d from rank %d\n", my_rank, 0);
      MPI_Probe(0, 0, MPI_COMM_WORLD, &mystat);
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

   free(sbuffer);
   sbuffer=NULL;

   free(rbuffer);
   rbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

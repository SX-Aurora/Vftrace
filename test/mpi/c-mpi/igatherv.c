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

   // require cmd-line argument
   if (argc < 2) {
      printf("./gatherv <msgsize in ints>\n");
      return 1;
   }

   int rootrank = 0;
   // allocating send/recv buffer
   int nints = atoi(argv[1])+my_rank;
   int *sbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {sbuffer[i]=my_rank;}
   int *rbuffer = NULL;

   // prepare special arrays for recvrank
   int *recvcounts = NULL;
   int *displs = NULL;
   if (my_rank == rootrank) {
      recvcounts = (int*) malloc(comm_size*sizeof(int));
      displs = (int*) malloc(comm_size*sizeof(int));
      int ntot = 0;
      for (int i=0; i<comm_size; i++) {
         recvcounts[i] = nints + i;
         displs[i] = ntot;
         ntot += recvcounts[i];
      }
      rbuffer = (int*) malloc(ntot*sizeof(int));
      for (int i=0; i<ntot; i++) {
         rbuffer[i] = -1;
      }
   }

   // Messaging
   MPI_Request myrequest;
   MPI_Igatherv(sbuffer, nints, MPI_INT,
                rbuffer, recvcounts, displs, MPI_INT, 
                rootrank, MPI_COMM_WORLD, &myrequest);
   if (my_rank == rootrank) {
      printf("Gathering messages from all ranks on rank %d\n", my_rank);
   }
   MPI_Status mystatus;
   MPI_Wait(&myrequest, &mystatus);

   // validate data
   bool valid_data = true;
   if (my_rank == rootrank) {
      for (int irank=0; irank<comm_size; irank++) {
         for (int i=0; i<recvcounts[irank]; i++) {
            if (rbuffer[i+displs[irank]] != irank) {
               printf("Rank %d received faulty data from rank %d\n", my_rank, irank);
               valid_data = false;
               break;
            }
         }
      }
      free(rbuffer);
      rbuffer=NULL;

      free(recvcounts);
      recvcounts = NULL;
      free(displs);
      displs = NULL;
   }

   free(sbuffer);
   sbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

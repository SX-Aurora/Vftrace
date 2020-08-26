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
      printf("./allgatherv <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]) + my_rank;
   int *sbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {sbuffer[i]=my_rank;}

   // prepare special arrays for recvrank
   int *recvcounts = (int*) malloc(comm_size*sizeof(int));
   int *displs = (int*) malloc(comm_size*sizeof(int));
   int ntot = 0;
   for (int i=0; i<comm_size; i++) {
      recvcounts[i] = nints - my_rank + i;
      displs[i] = ntot;
      ntot += recvcounts[i];
   }
   int *rbuffer = (int*) malloc(ntot*sizeof(int));
   for (int i=0; i<ntot; i++) {
      rbuffer[i] = -1;
   }

   // Messaging
   MPI_Request myrequest;
   MPI_Iallgatherv(sbuffer, nints, MPI_INT,
                   rbuffer, recvcounts, displs, MPI_INT,
                   MPI_COMM_WORLD, &myrequest);
   printf("Gathering messages from all ranks on all other ranks\n");
   MPI_Status mystatus;
   MPI_Wait(&myrequest, &mystatus);

   // validate data
   bool valid_data = true;
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

   free(sbuffer);
   sbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

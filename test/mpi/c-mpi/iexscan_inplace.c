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
      printf("./iexscan_inplace <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int *sbuffer = MPI_IN_PLACE;
   int *rbuffer = NULL;
   rbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {
      rbuffer[i] = my_rank;
   }

   // Messaging
   MPI_Request myrequest;
   MPI_Iexscan(sbuffer, rbuffer, nints, MPI_INT, 
               MPI_SUM, MPI_COMM_WORLD, &myrequest);
   printf("Scanning messages from all ranks on rank %d\n", my_rank);
   MPI_Status mystatus;
   MPI_Wait(&myrequest, &mystatus);

   // validate data
   bool valid_data = true;
   int refresult = ((my_rank-1)*my_rank)/2;
   for (int i=0; i<nints; i++) {
      if (rbuffer[i] != refresult) {
         printf("Rank %d received faulty data\n", my_rank);
         valid_data = false;
         break;
      }
   }

   free(rbuffer);
   rbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

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
      printf("./reduce_scatter_inplace <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int ntots = comm_size*nints+((comm_size-1)*comm_size)/2;
   int *recvcounts = (int*) malloc(comm_size*sizeof(int));
   for (int irank=0; irank<comm_size; irank++) {
      recvcounts[irank] = nints + irank;
   }

   int *rbuffer = (int*) malloc(ntots*sizeof(int));
   int idx = 0;
   for (int irank=0; irank<comm_size; irank++) {
      for (int i=0; i<recvcounts[irank]; i++) {
         rbuffer[idx]=irank;
         idx++;
      }
   }

   // Messaging cycle
   MPI_Reduce_scatter(MPI_IN_PLACE, rbuffer, recvcounts,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   printf("Reducing and scattering messages from all ranks to all ranks on rank %d\n", my_rank);

   // validate data
   bool valid_data = true;
   for (int i=0; i<recvcounts[my_rank]; i++) {
      if (rbuffer[i] != comm_size*my_rank) {
         printf("Rank %d received faulty data\n", my_rank);
         valid_data = false;
         break;
      }
   }

   free(rbuffer);
   rbuffer=NULL;

   free(recvcounts);
   recvcounts=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

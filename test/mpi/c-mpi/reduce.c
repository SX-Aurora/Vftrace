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
      printf("./reduce <msgsize in ints>\n");
      return 1;
   }

   int rootrank = 0;
   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int *sbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {sbuffer[i]=my_rank;}
   int *rbuffer = NULL;
   if (my_rank == rootrank) {
      rbuffer = (int*) malloc(nints*sizeof(int));
      for (int i=0; i<nints; i++) {
         rbuffer[i] = -1;
      }
   }

   // Messaging cycle
   MPI_Reduce(sbuffer, rbuffer, nints, MPI_INT,
              MPI_SUM, rootrank, MPI_COMM_WORLD);
   if (my_rank == rootrank) {
      printf("Reducing messages from all ranks on rank %d\n", my_rank);
   }

   // validate data
   bool valid_data = true;
   if (my_rank == rootrank) {
      int refresult = ((comm_size-1)*comm_size)/2;
      for (int i=0; i<nints; i++) {
         if (rbuffer[i] != refresult) {
            printf("Rank %d received faulty data\n", my_rank);
            valid_data = false;
            break;
         }
      }
      free(rbuffer);
      rbuffer=NULL;
   }

   free(sbuffer);
   sbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

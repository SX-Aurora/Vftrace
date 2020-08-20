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
      printf("./sendrecv_replace <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int* srbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {srbuffer[i]=my_rank;}

   // Messaging cycle
   MPI_Status mystat;
   int destrank = (my_rank + 1) % comm_size;
   int sourcerank = my_rank - 1;
   if (sourcerank < 0) {
      sourcerank = comm_size - 1;
   }

   MPI_Sendrecv_replace(srbuffer, nints, MPI_INT,
                        destrank, 0,
                        sourcerank, 0,
                        MPI_COMM_WORLD, &mystat);

   // validate data
   bool valid_data = true;
   for (int i=0; i<nints; i++) {
      if (srbuffer[i] != sourcerank) {
         printf("Rank %d received faulty data from rank %d\n", my_rank, destrank);
         valid_data = false;
         break;
      }
   }

   free(srbuffer);
   srbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

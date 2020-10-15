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
      printf("./bcast <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int* buffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {buffer[i]=my_rank;}

   // Messaging cycle
   int sendrank = 0;
   MPI_Bcast(buffer, nints, MPI_INT, sendrank, MPI_COMM_WORLD);
   if (my_rank == sendrank) {
      printf("Broadcasted message from rank %d\n", my_rank);
   }

   // validate data
   bool valid_data = true;
   for (int i=0; i<nints; i++) {
      if (buffer[i] != sendrank) {
         printf("Rank %d received faulty data from rank %d\n", my_rank, sendrank);
         valid_data = false;
         break;
      }
   }

   free(buffer);
   buffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

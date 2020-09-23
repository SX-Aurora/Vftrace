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
      printf("./allgather_inplace <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int *sbuffer = MPI_IN_PLACE;
   int *rbuffer = (int*) malloc(comm_size*nints*sizeof(int));
   for (int i=0; i<comm_size*nints; i++) {
      rbuffer[i] = my_rank;
   }

   // Messaging cycle
   MPI_Allgather(sbuffer, nints, MPI_INT,
                 rbuffer, nints, MPI_INT, 
                 MPI_COMM_WORLD);
   printf("Gathering messages from all ranks on rank %d\n", my_rank);

   // validate data
   bool valid_data = true;
   for (int irank=0; irank<comm_size; irank++) {
      for (int i=0; i<nints; i++) {
         if (rbuffer[i+irank*nints] != irank) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, irank);
            valid_data = false;
            break;
         }
      }
   }

   free(rbuffer);
   rbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

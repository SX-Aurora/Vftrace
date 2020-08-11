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
      printf("./put <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nbyte = atoi(argv[1]);
   int* srbuffer = malloc(nbyte);

   // open memory to remote memory access
   MPI_Win window;
   MPI_Win_create(srbuffer, nbyte, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &window);

   MPI_Win_fence(0, window);

   // Remote memory access
   if (my_rank == 0) {
      // send to every other rank
      for (int targetrank=1; targetrank<comm_size; targetrank++) {
         printf("Putting data remotely on rank %d\n", targetrank);
         MPI_Put(srbuffer, nbyte, MPI_BYTE, targetrank, 0, nbyte, MPI_BYTE, window);
      }
   }
   MPI_Win_fence(0, window);
   MPI_Win_free(&window);

   free(srbuffer);
   srbuffer=NULL;

   MPI_Finalize();

   return 0;
}

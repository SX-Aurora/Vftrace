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
      printf("./get <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int* srbuffer;
   if (my_rank == 0) {
      srbuffer = (int*) malloc((comm_size-1)*nints*sizeof(int));
      for (int i=0; i<(comm_size-1)*nints; i++) {srbuffer[i]=my_rank;}
   } else {
      srbuffer = (int*) malloc(nints*sizeof(int));
      for (int i=0; i<nints; i++) {srbuffer[i]=my_rank;}
   }

   // open memory to remote memory access
   MPI_Win window;
   MPI_Win_create(srbuffer, nints*sizeof(int), sizeof(int),
                  MPI_INFO_NULL, MPI_COMM_WORLD, &window);

   MPI_Win_fence(0, window);

   // Remote memory access
   bool valid_data = true;
   if (my_rank == 0) {
      // recv from every other rank
      MPI_Request *myrequests = (MPI_Request*) malloc((comm_size-1)*sizeof(MPI_Request));
      for (int targetrank=1; targetrank<comm_size; targetrank++) {
         printf("Collecting data remotely from rank %d\n", targetrank);
         int* rbuffptr = srbuffer+((targetrank-1)*nints);
         MPI_Rget(rbuffptr, nints, MPI_INT, targetrank, 0, nints,
                  MPI_INT, window, myrequests+targetrank-1);
      }
      MPI_Status *mystatuses = (MPI_Status*) malloc((comm_size-1)*sizeof(MPI_Status));
      MPI_Waitall(comm_size - 1, myrequests, mystatuses);
      free(myrequests);
      myrequests = NULL;
      free(mystatuses);
      mystatuses = NULL;
   }

   MPI_Win_fence(0, window);
   MPI_Win_free(&window);

   // validate data
   if (my_rank == 0) {
      for (int ipeer=1; ipeer<comm_size; ipeer++) {
         for (int i=0; i<nints; i++) {
            if (srbuffer[i+(ipeer-1)*nints] != ipeer) {
               printf("Rank %d received faulty data from rank %d\n", my_rank, ipeer);
               valid_data = false;
               break;
            }
         }
      }
   }

   free(srbuffer);
   srbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

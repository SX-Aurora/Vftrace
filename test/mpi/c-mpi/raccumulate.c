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
      printf("./raccumulate <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int* srbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {srbuffer[i]=my_rank;}

   // open memory to remote memory access
   MPI_Win window;
   MPI_Win_create(srbuffer, nints*sizeof(int), sizeof(int),
                  MPI_INFO_NULL, MPI_COMM_WORLD, &window);

   MPI_Win_fence(0, window);

   // Remote memory access
   if (my_rank == 0) {
      // send to every other rank
      for (int i=0; i<nints; i++) {srbuffer[i]=comm_size;}
      MPI_Request *myrequests = (MPI_Request*) malloc((comm_size-1)*sizeof(MPI_Request));
      for (int targetrank=1; targetrank<comm_size; targetrank++) {
         printf("Accumulating data remotely on rank %d\n", targetrank);
         MPI_Raccumulate(srbuffer, nints, MPI_INT, targetrank, 0, nints, MPI_INT, MPI_SUM, window, myrequests+targetrank-1);
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

   MPI_Barrier(MPI_COMM_WORLD);

   //validate data
   bool valid_data = true;
   for (int i=0; i<nints; i++) {
      if (srbuffer[i] != comm_size+my_rank) {
         printf("Rank %d received faulty data from rank 0\n", my_rank);
         valid_data = false;
         break;
      }
   }


   free(srbuffer);
   srbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

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
      printf("./fetch_and_op <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int *originbuffer;
   int *resultbuffer;
   int *targetbuffer;
   if (my_rank == 0) {
      originbuffer = (int*) malloc(nints*sizeof(int));
      for (int i=0; i<nints; i++) {originbuffer[i]=my_rank;}
      resultbuffer = (int*) malloc(nints*sizeof(int));
      for (int i=0; i<nints; i++) {resultbuffer[i]=0;}
   } else {
      targetbuffer = (int*) malloc(nints*sizeof(int));
      for (int i=0; i<nints; i++) {targetbuffer[i]=my_rank;}
   }

   // open memory to remote memory access
   MPI_Win window;
   MPI_Win_create(targetbuffer, nints*sizeof(int), sizeof(int),
                  MPI_INFO_NULL, MPI_COMM_WORLD, &window);

   MPI_Win_fence(0, window);

   // Remote memory access
   if (my_rank == 0) {
      for (int irank=1; irank<comm_size; irank++) {
         // Origin stays unchanged
         // Resultbuffer gets a copy of the target buffer from remote process
         // The remote target buffer gets the sum of origin+itself
         MPI_Fetch_and_op(originbuffer, resultbuffer, MPI_INT,
                          irank, 0, MPI_SUM, window);
         // copy the resultbuffer to the origin buffer
         for (int i=0; i<nints; i++) {originbuffer[i]+=resultbuffer[i];}
      }
   }
   
   MPI_Win_fence(0, window);
   MPI_Win_free(&window);

   // validate data
   bool valid_data = true;
   if (my_rank == 0) {
      // contents of origin buffer should be the summ of all ranks
      for (int i=0; i<nints; i++) {
         int refresult;
         if (i==0) {
            refresult = (comm_size*(comm_size-1))/2;
         } else {
            refresult = my_rank;
         }
         if (originbuffer[i] != refresult) {
            printf("Rank %d received faulty data\n", my_rank);
            valid_data = false;
            break;
         }
      }
      // contents of Result buffer should be the largest rank
      for (int i=0; i<nints; i++) {
         int refresult;
         if (i==0) {
            refresult = comm_size-1;
         } else {
            refresult = 0;
         }
         if (resultbuffer[i] != refresult) {
            printf("Rank %d received faulty data\n", my_rank);
            valid_data = false;
            break;
         }
      }

      free(originbuffer);
      originbuffer=NULL;
      free(resultbuffer);
      resultbuffer=NULL;
   } else {
      // contents of target buffer should be the sum of all ranks up to this one
      for (int i=0; i<nints; i++) {
         int refresult;
         if (i==0) {
            refresult = (my_rank*(my_rank+1))/2;
         } else {
            refresult = my_rank;
         }
         if (targetbuffer[i] != refresult) {
            printf("Rank %d received faulty data\n", my_rank);
            valid_data = false;
            break;
         }
      }
      free(targetbuffer);
      targetbuffer=NULL;
   }

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

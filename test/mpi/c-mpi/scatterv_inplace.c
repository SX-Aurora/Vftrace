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
      printf("./scatterv_inplace <msgsize in ints>\n");
      return 1;
   }

   int rootrank = 0;
   // allocating send/recv buffer
   int nints = atoi(argv[1]) + my_rank;
   int *rbuffer = NULL;
   int *sbuffer = NULL;

   // prepare special arrays for sendrank
   int *sendcounts = NULL;
   int *displs = NULL;
   if (my_rank == rootrank) {
      sendcounts = (int*) malloc(comm_size*sizeof(int));
      displs = (int*) malloc(comm_size*sizeof(int));
      int ntot = 0;
      for (int i=0; i<comm_size; i++) {
         sendcounts[i] = nints + i;
         displs[i] = ntot;
         ntot += sendcounts[i];
      }
      sbuffer = (int*) malloc(ntot*sizeof(int));
      for (int irank=0; irank<comm_size; irank++) {
         for (int i=0; i<sendcounts[irank]; i++) {
            sbuffer[i+displs[irank]] = irank;
         }
      }
      rbuffer = MPI_IN_PLACE;
   } else {
      rbuffer = (int*) malloc(nints*sizeof(int));
      for (int i=0; i<nints; i++) {rbuffer[i]=-1;}
   }

   // Messaging
   MPI_Scatterv(sbuffer, sendcounts, displs, MPI_INT,
                rbuffer, nints, MPI_INT, 
                rootrank, MPI_COMM_WORLD);
   if (my_rank == rootrank) {
      printf("Scattering messages to all ranks from rank %d\n", my_rank);
   }

   // validate data
   bool valid_data = true;
   if (my_rank == rootrank) {
      for (int i=0; i<nints; i++) {
         if (sbuffer[i+displs[my_rank]] != my_rank) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, rootrank);
            valid_data = false;
            break;
         }
      }
   } else {
      for (int i=0; i<nints; i++) {
         if (rbuffer[i] != my_rank) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, rootrank);
            valid_data = false;
            break;
         }
      }
   }

   if (my_rank == rootrank) {
      free(sbuffer);
      sbuffer=NULL;

      free(sendcounts);
      sendcounts=NULL;

      free(displs);
      displs = NULL;
   } else {
      free(rbuffer);
      rbuffer=NULL;
   }

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

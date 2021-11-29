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
      printf("./neighbor_allgatherv_cart_periodic <msgsize in ints>\n");
      return 1;
   }

   // requires precicely 4 processes
   if (comm_size != 4) {
      printf("requires precicely 4 processes. Start with -np 4!\n");
      return 1;
   }

   // Create the cartesian communicator
   MPI_Comm comm_cart;
   int ndims = 3;
   int dims[3] = {2,2,1};
   // determine own coordinates
   int periods[3] = {true,true,true};
   int reorder = false;
   MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);

   int nneighbors = 2*ndims;
   // fill the neighbor list
   int neighbors[6] = {0,0,0,0,0,0};
   switch(my_rank) {
      case 0:
         neighbors[0] = 2;
         neighbors[1] = 2;
         neighbors[2] = 1;
         neighbors[3] = 1;
         break;
      case 1:
         neighbors[0] = 3;
         neighbors[1] = 3;
         neighbors[4] = 1;
         neighbors[5] = 1;
         break;
      case 2:
         neighbors[2] = 3;
         neighbors[3] = 3;
         neighbors[4] = 2;
         neighbors[5] = 2;
         break;
      case 3:
         neighbors[0] = 1;
         neighbors[1] = 1;
         neighbors[2] = 2;
         neighbors[3] = 2;
         neighbors[4] = 3;
         neighbors[5] = 3;
         break;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]) + my_rank;
   int *sbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {sbuffer[i]=my_rank;}

   int *recvcounts = (int*) malloc(nneighbors*sizeof(int));
   int *displs = (int*) malloc(nneighbors*sizeof(int));
   int ntot = 0;
   for (int ineighbor=0; ineighbor<nneighbors; ineighbor++) {
         recvcounts[ineighbor] = nints - my_rank + neighbors[ineighbor];
         displs[ineighbor] = ntot;
         ntot += recvcounts[ineighbor];
         if (neighbors[ineighbor] == -1) {
            recvcounts[ineighbor] = 0;
         }

   }
   int *rbuffer = (int*) malloc(ntot*sizeof(int));
   for (int i=0; i<ntot; i++) {rbuffer[i]=-1;}

   MPI_Neighbor_allgatherv(sbuffer, nints, MPI_INT,
                           rbuffer, recvcounts, displs, MPI_INT,
                           comm_cart);

   // validate data
   bool valid_data = true;
   for (int ineighbor=0; ineighbor<nneighbors; ineighbor++) {
      int refval = -1;
      if (neighbors[ineighbor] >= 0) {
         refval = neighbors[ineighbor];
      }
      for (int i=0; i<recvcounts[ineighbor]; i++) {
         if (rbuffer[i+displs[ineighbor]] != refval) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, refval);
            valid_data = false;
            break;
         }
      }
   }
   free(recvcounts);
   recvcounts = NULL;
   free(displs);
   displs = NULL;
   free(rbuffer);
   rbuffer=NULL;
   free(sbuffer);
   sbuffer=NULL;

   MPI_Comm_free(&comm_cart);
   MPI_Finalize();

   return valid_data ? 0 : 1;
}

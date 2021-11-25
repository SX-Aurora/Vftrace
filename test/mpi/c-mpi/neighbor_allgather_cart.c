#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

#include "cart_neighbor_ranks.h"

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
      printf("./neighbor_allgather_cart <msgsize in ints>\n");
      return 1;
   }

   // Create the cartesian communicator
   MPI_Comm comm_cart;
   int ndims = 3;
   int *dims = (int*) malloc(ndims*sizeof(int));
   int *periods = (int*) malloc(ndims*sizeof(int));
   int reorder = false;
   for (int idim=0; idim<ndims; idim++) {
      dims[idim] = 0;
      periods[idim] = false;
   }
   MPI_Dims_create(comm_size, ndims, dims);
   MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);
   free(dims);
   dims = NULL;
   free(periods);
   periods = NULL;

   // allocating send/recv buffer
   int nneighbors = 2*ndims;
   int nints = atoi(argv[1]);
   int *sbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {sbuffer[i]=my_rank;}
   int *rbuffer = (int*) malloc(nints*nneighbors*sizeof(int));
   for (int i=0; i<nints*nneighbors; i++) {rbuffer[i]=-1;}

   MPI_Neighbor_allgather(sbuffer, nints, MPI_INT,
                          rbuffer, nints, MPI_INT,
                          comm_cart);

   // validate data
   bool valid_data = true;
   int *list_of_neighbor_ranks;
   cart_neighbor_ranks(comm_cart,
                       &nneighbors,
                       &list_of_neighbor_ranks);
   for (int ineighbor=0; ineighbor<nneighbors; ineighbor++) {
      int refval = -1;
      if (list_of_neighbor_ranks[ineighbor] >= 0) {
         refval = list_of_neighbor_ranks[ineighbor];
      }
      for (int i=0; i<nints; i++) {
         if (rbuffer[i+ineighbor*nints] != refval) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, refval);
            valid_data = false;
            break;
         }
      }
   }
   free(list_of_neighbor_ranks);
   list_of_neighbor_ranks = NULL;
   free(rbuffer);
   rbuffer=NULL;
   free(sbuffer);
   sbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}

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
      printf("./ineighbor_allgatherv_graph <msgsize in ints>\n");
      return 1;
   }

   // requires precicely 4 processes
   if (comm_size != 4) {
      printf("requires precicely 4 processes. Start with -np 4!\n");
      return 1;
   }

   // Create the cartesian communicator
   MPI_Comm comm_graph;
   const int nnodes = 4;
   int index[4] = {3,6,9,11};
   int edges[11] = {1,1,2,0,0,2,0,1,3,2,3};
   int reorder = false;
   MPI_Graph_create(MPI_COMM_WORLD, nnodes, index, edges, reorder, &comm_graph);

   int nneighbors;
   MPI_Graph_neighbors_count(comm_graph, my_rank, &nneighbors);
   int *neighbors = (int*) malloc(nneighbors*sizeof(int));
   MPI_Graph_neighbors(comm_graph, my_rank, nneighbors, neighbors);

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
   }
   int *rbuffer = (int*) malloc(ntot*sizeof(int));
   for (int i=0; i<ntot; i++) {rbuffer[i]=-1;}

   MPI_Request request;
   MPI_Ineighbor_allgatherv(sbuffer, nints, MPI_INT,
                            rbuffer, recvcounts, displs, MPI_INT,
                            comm_graph, &request);
   MPI_Status status;
   MPI_Wait(&request, &status);

   // validate data
   bool valid_data = true;
   for (int ineighbor=0; ineighbor<nneighbors; ineighbor++) {
      int refval = neighbors[ineighbor];
      for (int i=0; i<recvcounts[ineighbor]; i++) {
         if (rbuffer[i+displs[ineighbor]] != refval) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, refval);
            valid_data = false;
            break;
         }
      }
   }
   free(neighbors);
   neighbors = NULL;
   free(recvcounts);
   recvcounts = NULL;
   free(displs);
   displs = NULL;
   free(rbuffer);
   rbuffer=NULL;
   free(sbuffer);
   sbuffer=NULL;

   MPI_Comm_free(&comm_graph);
   MPI_Finalize();

   return valid_data ? 0 : 1;
}
